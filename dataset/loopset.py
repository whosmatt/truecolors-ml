"""Period-labelled corpus from the BPM-verified drum loops.

20.2 hours of real audio that cannot supervise beats: a duration-verified tempo
fixes the *period* exactly but says nothing about the *phase*, and deriving phase
from the audio was measured at 20 ms median error with 17% off-beat locks.

Period alone is still worth having, and it is exactly what stage 2 gets wrong
most often. So these takes carry an exact period label and masked beat and drum
labels (-1), to be trained jointly with the rendered clips that do have a grid.

    python -m dataset.loopset --out data/loops --limit 4000
    python -m dataset.loopset --out data/melodic --melodic
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

from . import features, gridset, render

MASK = -1.0


def len_err(row: dict) -> float:
    b = row["duration_s"] * row["bpm"] / 60.0
    return abs(b - round(b)) / max(round(b), 1)


def melodic_pool(manifest: str) -> list[dict]:
    """Untagged is not drum-free (see render.backing_pool): this is music without
    a drum label, not music without drums."""
    out = []
    with open(manifest) as fh:
        fh.readline()
        for line in fh:
            r = json.loads(line)
            if (r["kind"] == "loop" and r.get("bpm") and r.get("readable")
                    and not r.get("discard")
                    and "Drums" not in set(r.get("families") or ())
                    and not render.matches_pack(r["path"])):
                out.append(r)
    out.sort(key=lambda r: r["file_id"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", type=Path, default=Path("data/loops"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-hicut", action="store_true", help="build without the 4 kHz hi-cut")
    ap.add_argument("--grid", action="store_true",
                    help="keep beat labels anchored at the file start instead of masking them")
    ap.add_argument("--max-len-err", type=float, default=None,
                    help="drop loops whose length is further than this fraction from whole beats")
    ap.add_argument("--melodic", action="store_true",
                    help="loops without a Drums tag instead; the backing-bed packs are "
                         "excluded, since rendered clips already train on them")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pool = melodic_pool(a.manifest) if a.melodic else gridset.loop_pool(a.manifest)
    if a.max_len_err is not None:
        pool = [r for r in pool if len_err(r) <= a.max_len_err]
    if a.limit:
        pool = pool[: a.limit]
    acc = {s: {k: [] for k in ("X", "y", "off", "beat", "beat_off", "period",
                               "take", "notch")} for s, _ in gridset.SPLITS}
    skip = int(gridset.SETTLE_S * SAMPLE_RATE / BLOCK_SAMPLES)
    n, t0 = 0, time.time()

    for i, row in enumerate(pool):
        for rep in range(a.repeats):
            rng = np.random.default_rng(row["file_id"] + 7919 * rep)
            notch = features.NOTCH_HZ[(i + rep) % len(features.NOTCH_HZ)]
            r = gridset.take(row, rng, notch, hicut=not a.no_hicut)
            if r is None:
                continue
            X = r[0][skip:]
            if len(X) < 300:
                continue
            m = len(X)
            beat_blocks = 60.0 / row["bpm"] * SAMPLE_RATE / BLOCK_SAMPLES
            d = acc[gridset.split_of(row)]
            d["X"].append(X)
            d["y"].append(np.full((m, 4), MASK, dtype=np.float32))
            d["off"].append(np.full((m, 3), MASK, dtype=np.float32))
            if a.grid:
                d["beat"].append(r[1][skip:])
                d["beat_off"].append(r[2][skip:])
            else:
                d["beat"].append(np.full((m, 1), MASK, dtype=np.float32))
                d["beat_off"].append(np.full((m, 1), MASK, dtype=np.float32))
            d["period"].append(np.full((m, 1), beat_blocks, dtype=np.float32))
            d["take"].append(np.full(m, n, dtype=np.int32))
            d["notch"].append(np.full(m, notch, dtype=np.int16))
            n += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pool)}  {n} takes  {time.time()-t0:.0f}s", flush=True)

    meta = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": ("BPM-verified loops without a Drums tag" if a.melodic
                       else "BPM-verified drum loops") + ", period label only",
            "masked": ["y", "off"] + ([] if a.grid else ["beat", "beat_off"]),
            "grid": "file start" if a.grid else None, "max_len_err": a.max_len_err, "takes": n,
            **features.spec(hicut=not a.no_hicut)}
    for s, _ in gridset.SPLITS:
        d = acc[s]
        if not d["X"]:
            continue
        arrs = {k: np.concatenate(v) for k, v in d.items()}
        np.savez(a.out / f"{s}.npz", **arrs)
        meta.setdefault("splits", {})[s] = {"blocks": int(len(arrs["X"])),
                                            "takes": int(len(np.unique(arrs["take"])))}
        print(f"{s:6} {len(arrs['X']):9,} blocks  {len(np.unique(arrs['take'])):5,} takes")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
