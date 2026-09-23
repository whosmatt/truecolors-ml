"""Assemble the training set: render every drum clip, featurise, label.

    python -m dataset.build --out data/features

Splits are by clip, not by block. A clip rendered at three notch settings shares
its groove and its kit, so letting those land in different splits would leak the
answer. Hashing the clip name also keeps the split stable as clips are added.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE as SR

from . import alc, augment, features, render
from .ableton import DETECTION_CLASSES, Library

SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def split_of(name: str) -> str:
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") / 2**32
    acc = 0.0
    for label, frac in SPLITS:
        acc += frac
        if h < acc:
            return label
    return SPLITS[-1][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=render.AGC_SETTLE_S + render.KEEP_S)
    ap.add_argument("--repeats", type=int, default=1,
                    help="renders per clip, each with a different kit")
    ap.add_argument("--no-backing", action="store_true")
    ap.add_argument("--no-augment", action="store_true",
                    help="skip the IR and whine; dry renders only")
    ap.add_argument("--ir", type=Path, default=augment.IR_PATH)
    ap.add_argument("--no-hicut", action="store_true", help="build without the 4 kHz hi-cut")
    ap.add_argument("--db", type=Path, default=None,
                    help="Live index to read clips from; a saved copy pins the clip set, "
                         "since the manifest only covers one-shots and loops")
    ap.add_argument("--laser-off", action="store_true",
                    help="inject the quiet mic floor instead of coil whine")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    lib = Library(a.db) if a.db else Library()
    pools = render.pools_from_manifest(a.manifest)
    beds = {} if a.no_backing else render.backing_pool(a.manifest)
    if not a.no_augment:
        augment.IR_PATH = a.ir
        print(f"augmenting with {a.ir} + measured coil whine per PWM setting")
    print("one-shot pools:", {k: len(v) for k, v in sorted(pools.items())})
    print("backing tempos:", len(beds))

    ids = alc.drum_clips(lib)[: a.limit or None]
    acc = {s: {"X": [], "y": [], "off": [], "take": [], "notch": [], "beat": [],
               "beat_off": [], "downbeat": [], "downbeat_off": [], "phrase": [],
               "music": []} for s, _ in SPLITS}
    takes, skipped, t0 = 0, 0, time.time()
    bed_cache: dict[str, np.ndarray] = {}

    for i, fid in enumerate(ids):
        clip = alc.parse(lib.path_of(fid), lib.name_of(fid))
        if clip is None or not clip.notes:
            skipped += 1
            continue
        split = split_of(clip.name)
        for rep in range(a.repeats):
            # A fresh kit and bed per repeat: the same groove voiced by different
            # drums is the cheapest source of variety, and the model overfits
            # (train beat-loss 0.45 against 0.58 val) well before it runs out of
            # compute budget.
            kit = render.Kit(pools, seed=f"{clip.name}#{rep}")
            rng = np.random.default_rng(fid + 7919 * rep)

            bed = None
            cands = beds.get(int(round(clip.tempo)), [])
            if cands:
                row = cands[int(rng.integers(len(cands)))]
                if row["path"] not in bed_cache:
                    if len(bed_cache) > 64:  # beds are ~1.5 MB each
                        bed_cache.clear()
                    loaded = render._load(row["path"])
                    bed_cache[row["path"]] = (loaded if loaded is not None
                                              else np.zeros(0, np.float32))
                if bed_cache[row["path"]].size:
                    bed = (bed_cache[row["path"]], row["name"])

            r = render.render(clip, kit, classes=DETECTION_CLASSES, backing=bed,
                              min_seconds=a.seconds)
            # Convolution is linear and notch-independent: once per render.
            wet = None if a.no_augment else augment.apply_ir(r.audio)
            for notch in features.NOTCH_HZ:
                dbfs = float(rng.uniform(*render.LEVEL_DBFS))
                if a.no_augment:
                    pcm = r.to_int16(dbfs)
                else:
                    y = augment.finish(wet, None if a.laser_off else notch, rng, dbfs=dbfs)
                    pcm = (y * 32767.0).astype(np.int16)
                X = features.featurise(pcm, hicut=not a.no_hicut)
                y, off = features.label_blocks(
                    len(X), r.onsets, r.classes, len(DETECTION_CLASSES)
                )
                # Beat grid, exact: the clip's MIDI puts beat 0 at its start and
                # the take is whole repeats of the clip, so beats run at 60/tempo
                # throughout. Deriving it from real loops instead gives 20 ms
                # median error and 17% off-beat locks.
                beat_s = 60.0 / clip.tempo
                n_beats = len(X) * BLOCK_SAMPLES / SR / beat_s
                bt = np.arange(int(n_beats) + 1) * beat_s * SR
                beat, beat_off = features.label_blocks(
                    len(X), bt.astype(np.int64), np.zeros(len(bt), dtype=np.int8), 1
                )
                # Downbeats, assuming 4/4: 99.9% of clip loop lengths are a whole
                # number of 4-beat bars, and none are a multiple of 3 but not 4.
                db = bt[::4]
                downbeat, downbeat_off = features.label_blocks(
                    len(X), db.astype(np.int64), np.zeros(len(db), dtype=np.int8), 1
                )
                # Length of the repeating pattern, in beats. The firmware asked
                # for this: the grid has no concept of a loop without it.
                phrase = np.full((len(X), 1), float(round(clip.beats)), dtype=np.float32)
                # Explicit "nothing here" channel: without it the only negative
                # signal is the absence of a positive.
                none = (y.sum(axis=1, keepdims=True) == 0).astype(np.float32)
                y = np.concatenate([y, none], axis=1)
                # Drop the settle window: the AGC has not converged there.
                skip = int(render.AGC_SETTLE_S * SR / BLOCK_SAMPLES)
                X, y, off = X[skip:], y[skip:], off[skip:]
                beat, beat_off = beat[skip:], beat_off[skip:]
                downbeat, downbeat_off = downbeat[skip:], downbeat_off[skip:]
                phrase = phrase[skip:]
                if not len(X):
                    continue
                d = acc[split]
                d["X"].append(X)
                d["y"].append(y)
                d["off"].append(off)
                d["beat"].append(beat)
                d["beat_off"].append(beat_off)
                d["downbeat"].append(downbeat)
                d["downbeat_off"].append(downbeat_off)
                d["phrase"].append(phrase)
                d["music"].append(np.ones((len(X), 1), dtype=np.float32))
                d["take"].append(np.full(len(X), takes, dtype=np.int32))
                d["notch"].append(np.full(len(X), notch, dtype=np.int16))
                takes += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ids)} clips  {takes} takes  {time.time()-t0:.0f}s", flush=True)

    meta = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "classes": list(DETECTION_CLASSES),
        "beat_labels": True,
        "downbeat_labels": True,
        "music_labels": True,
        "takes": takes,
        "clips_skipped": skipped,
        "seconds_per_take": a.seconds,
        "repeats": a.repeats,
        "backing": not a.no_backing,
        "agc_settle_s": render.AGC_SETTLE_S,
        "level_dbfs": list(render.LEVEL_DBFS),
        "augmented": not a.no_augment,
        "laser_off": a.laser_off,
        "ir": None if a.no_augment else str(a.ir),
        "ir_sha256": None if a.no_augment or not a.ir.exists()
                     else hashlib.sha256(a.ir.read_bytes()).hexdigest(),
        **features.spec(hicut=not a.no_hicut),
    }
    for s, _ in SPLITS:
        d = acc[s]
        if not d["X"]:
            continue
        arrs = {k: np.concatenate(v) for k, v in d.items()}
        np.savez(a.out / f"{s}.npz", **arrs)
        pos = arrs["y"].sum(axis=0).astype(int)
        meta.setdefault("splits", {})[s] = {
            "blocks": int(len(arrs["X"])),
            "takes": int(len(np.unique(arrs["take"]))),
            "positives": dict(zip(DETECTION_CLASSES, pos.tolist())),
        }
        print(f"{s:6} {len(arrs['X']):8} blocks  positives {dict(zip(DETECTION_CLASSES, pos.tolist()))}")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta.get("splits", {}), indent=2))


if __name__ == "__main__":
    main()
