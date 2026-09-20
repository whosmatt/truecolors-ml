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
    ap.add_argument("--no-backing", action="store_true")
    ap.add_argument("--no-augment", action="store_true",
                    help="skip the IR and whine; dry renders only")
    ap.add_argument("--ir", type=Path, default=augment.IR_PATH)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    lib = Library()
    pools = render.pools_from_manifest(a.manifest)
    beds = {} if a.no_backing else render.backing_pool(a.manifest)
    if not a.no_augment:
        augment.IR_PATH = a.ir
        print(f"augmenting with {a.ir} + measured coil whine per PWM setting")
    print("one-shot pools:", {k: len(v) for k, v in sorted(pools.items())})
    print("backing tempos:", len(beds))

    ids = alc.drum_clips(lib)[: a.limit or None]
    acc = {s: {"X": [], "y": [], "off": [], "take": [], "notch": []} for s, _ in SPLITS}
    takes, skipped, t0 = 0, 0, time.time()
    bed_cache: dict[str, np.ndarray] = {}

    for i, fid in enumerate(ids):
        clip = alc.parse(lib.path_of(fid), lib.name_of(fid))
        if clip is None or not clip.notes:
            skipped += 1
            continue
        split = split_of(clip.name)
        kit = render.Kit(pools, seed=clip.name)

        bed = None
        cands = beds.get(int(round(clip.tempo)), [])
        if cands:
            row = cands[int(np.random.default_rng(fid).integers(len(cands)))]
            if row["path"] not in bed_cache:
                if len(bed_cache) > 64:  # beds are ~1.5 MB each; keep a small pool
                    bed_cache.clear()
                loaded = render._load(row["path"])
                bed_cache[row["path"]] = loaded if loaded is not None else np.zeros(0, np.float32)
            if bed_cache[row["path"]].size:
                bed = (bed_cache[row["path"]], row["name"])

        rng = np.random.default_rng(fid)
        # One render, featurised once per notch: the audio is identical across
        # notch settings, only the front end's comb differs.
        r = render.render(
            clip, kit, classes=DETECTION_CLASSES, backing=bed, min_seconds=a.seconds
        )
        # Convolution is linear and notch-independent, so it happens once per
        # clip rather than once per take.
        wet = None if a.no_augment else augment.apply_ir(r.audio)
        for notch in features.NOTCH_HZ:
            dbfs = float(rng.uniform(*render.LEVEL_DBFS))
            if a.no_augment:
                pcm = r.to_int16(dbfs)
            else:
                # Coil whine for this PWM setting, at its measured absolute level.
                y = augment.finish(wet, notch, rng, dbfs=dbfs)
                pcm = (y * 32767.0).astype(np.int16)
            X = features.featurise(pcm, notch)
            y, off = features.label_blocks(
                len(X), r.onsets, r.classes, len(DETECTION_CLASSES)
            )
            # Drop the settle window: the AGC has not converged there, so those
            # blocks do not look like anything the device would see in steady state.
            skip = int(render.AGC_SETTLE_S * SR / BLOCK_SAMPLES)
            X, y, off = X[skip:], y[skip:], off[skip:]
            if not len(X):
                continue
            d = acc[split]
            d["X"].append(X)
            d["y"].append(y)
            d["off"].append(off)
            d["take"].append(np.full(len(X), takes, dtype=np.int32))
            d["notch"].append(np.full(len(X), notch, dtype=np.int16))
            takes += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ids)} clips  {takes} takes  {time.time()-t0:.0f}s", flush=True)

    meta = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "classes": list(DETECTION_CLASSES),
        "takes": takes,
        "clips_skipped": skipped,
        "seconds_per_take": a.seconds,
        "backing": not a.no_backing,
        "agc_settle_s": render.AGC_SETTLE_S,
        "level_dbfs": list(render.LEVEL_DBFS),
        "augmented": not a.no_augment,
        "ir": None if a.no_augment else str(a.ir),
        "ir_sha256": None if a.no_augment or not a.ir.exists()
                     else hashlib.sha256(a.ir.read_bytes()).hexdigest(),
        **features.spec(),
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
