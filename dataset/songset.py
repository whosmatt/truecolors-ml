"""Training corpus from hand-checked song grids (songs/labels, see dataset.songgrid).

Whole songs, beat-labelled end to end from the confirmed constant-tempo grid,
so the sections the model missed are labelled too. Spans the labeller excluded
are masked. Songs rejected as variable tempo or non-rhythmic are still music:
they are kept as music positives with the beat masked.

`locked` marks blocks in sections where the model held its own lock when the
grid was fitted, so training with and without the rest can be compared.

Labels are beat times on the audio file; features are rebuilt here through the
current IR and front end, so an FE or IR change needs a rebuild, not a relabel.

    python -m dataset.songset --out data/songs
    python -m dataset.songset --out data/songs_auto --auto   # auto 'accept' fits, unchecked
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

from . import augment, features, gridset, render
from .songgrid import SONGS, decode

MASK = -1.0


def split_of(song_id: str) -> str:
    h = int.from_bytes(hashlib.sha256(song_id.encode()).digest()[:4], "big") / 2**32
    acc = 0.0
    for label, frac in gridset.SPLITS:
        acc += frac
        if h < acc:
            return label
    return gridset.SPLITS[-1][0]


def labels(auto: bool) -> list[dict]:
    out = []
    if auto:
        for p in sorted((SONGS / "auto").glob("*.json")):
            d = json.loads(p.read_text())
            if d["verdict"]["auto"] == "accept":
                out.append({"id": d["id"], "status": "confirmed", "bpm": d["fit"]["bpm"],
                            "t0_s": d["fit"]["t0_s"], "sections": []})
    else:
        for p in sorted((SONGS / "labels").glob("*.json")):
            d = json.loads(p.read_text())
            if d["status"] in ("confirmed", "variable_tempo", "non_rhythmic"):
                out.append(d)
    return out


def grid_labels(n: int, bpm: float, t0: float, excluded) -> tuple[np.ndarray, np.ndarray]:
    beat = np.zeros((n, 1), np.float32)
    frac = np.zeros((n, 1), np.float32)
    T = 60.0 / bpm * SAMPLE_RATE
    pos = (t0 * SAMPLE_RATE) % T
    b = (pos + np.arange(int((n * BLOCK_SAMPLES - pos) / T) + 1) * T) / BLOCK_SAMPLES
    b = b[b < n]
    beat[b.astype(int), 0] = 1.0
    frac[b.astype(int), 0] = b % 1.0
    for s, e in excluded:
        sl = slice(int(s * SAMPLE_RATE / BLOCK_SAMPLES), int(np.ceil(e * SAMPLE_RATE / BLOCK_SAMPLES)))
        beat[sl], frac[sl] = MASK, MASK
    return beat, frac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/songs"))
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="renders per song, cycling whine and level")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation fold as test (dataset.songfolds) instead of the hash split")
    a = ap.parse_args()
    splitter = split_of
    if a.fold is not None:
        from . import songfolds
        folds = songfolds.load()
        splitter = lambda k: songfolds.split_for(k, a.fold, folds)
        (a.out / "folds.json").parent.mkdir(parents=True, exist_ok=True)
        (a.out / "folds.json").write_text(json.dumps(folds))
    a.out.mkdir(parents=True, exist_ok=True)
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    pool = labels(a.auto)
    skip = int(gridset.SETTLE_S * SAMPLE_RATE / BLOCK_SAMPLES)
    keys = ("X", "y", "off", "beat", "beat_off", "period", "music", "locked", "take", "notch")
    acc = {s: {k: [] for k in keys} for s, _ in gridset.SPLITS}
    n, t_start = 0, time.time()
    for i, lab in enumerate(pool):
        m = decode(SONGS / index[lab["id"]]["audio"])
        wet = augment.apply_ir(m, np.random.default_rng(int.from_bytes(lab["id"].encode()[:4], "big") + 104729))
        for rep in range(a.repeats):
            rng = np.random.default_rng(int.from_bytes(lab["id"].encode()[:4], "big") + 7919 * rep)
            notch = features.NOTCH_CYCLE[(i + rep) % len(features.NOTCH_CYCLE)]
            y = augment.finish(wet, notch, rng, dbfs=float(rng.uniform(*render.LEVEL_DBFS)))
            X = features.featurise((y * 32767.0).astype(np.int16), hicut=True)
            L = len(X)
            if lab["status"] == "confirmed":
                ex = [(s["start_s"], s["end_s"]) for s in lab.get("sections") or ()
                      if not s.get("include", True)]
                beat, frac = grid_labels(L, lab["bpm"], lab["t0_s"], ex)
                period = np.full((L, 1), 60.0 / lab["bpm"] * SAMPLE_RATE / BLOCK_SAMPLES, np.float32)
            else:
                beat = frac = period = np.full((L, 1), MASK, np.float32)
            locked = np.zeros((L, 1), np.float32)
            auto_path = SONGS / "auto" / f"{lab['id']}.json"
            if auto_path.exists():
                for sec in json.loads(auto_path.read_text())["sections"]:
                    if sec["solid"]:
                        locked[int(sec["start_s"] / (BLOCK_SAMPLES / SAMPLE_RATE)):
                               int(sec["end_s"] / (BLOCK_SAMPLES / SAMPLE_RATE))] = 1.0
            d = acc[splitter(lab["id"])]
            d["X"].append(X[skip:])
            d["y"].append(np.full((L - skip, 4), MASK, np.float32))
            d["off"].append(np.full((L - skip, 3), MASK, np.float32))
            d["beat"].append(beat[skip:])
            d["beat_off"].append(frac[skip:])
            d["period"].append(period[skip:])
            d["music"].append(np.ones((L - skip, 1), np.float32))
            d["locked"].append(locked[skip:])
            d["take"].append(np.full(L - skip, n, np.int32))
            d["notch"].append(np.full(L - skip, notch, np.int16))
            n += 1
        print(f"  {i+1}/{len(pool)} {lab['status']:15} {index[lab['id']]['title'][:40]}"
              f"  {time.time()-t_start:.0f}s", flush=True)

    meta = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "songs, " + ("auto-accepted grids, unchecked" if a.auto
                                   else "hand-checked constant-tempo grids"),
            "masked": ["y", "off"], "takes": n, "repeats": a.repeats, "fold": a.fold,
            **features.spec(hicut=True)}
    for s, _ in gridset.SPLITS:
        d = acc[s]
        if not d["X"]:
            continue
        arrs = {k: np.concatenate(v) for k, v in d.items()}
        np.savez(a.out / f"{s}.npz", **arrs)
        meta.setdefault("splits", {})[s] = {
            "blocks": int(len(arrs["X"])), "takes": int(len(np.unique(arrs["take"]))),
            "beats": int((arrs["beat"] > 0).sum()), "masked": int((arrs["beat"] < 0).sum())}
        print(f"{s:6} {len(arrs['X']):9,} blocks  {meta['splits'][s]['takes']:4} takes")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
