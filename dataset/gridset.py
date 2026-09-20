"""Beat-labelled corpus from real drum loops.

Approach 2 predicts the grid itself rather than drum hits. The label is "a beat
falls in this block", which is what stage 2 actually consumes — the lights care
about beats, not about whether a hit was a kick.

The supervision is free and exact: a loop with a duration-verified tempo is cut
sample-accurately to whole bars, so its start is a downbeat and every beat sits
at a known multiple of 60/bpm from it. Tiling from a random offset moves the grid
by a known amount, so the phase is never trivially zero.

    python -m dataset.gridset --out data/grid --limit 4000
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

from . import augment, features, render

TAKE_S = 16.0
SETTLE_S = 4.0
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def split_of(row: dict) -> str:
    """Stable per-loop split, so the same loops stay held out across approaches."""
    h = int.from_bytes(hashlib.sha256(row["path"].encode()).digest()[:4], "big") / 2**32
    acc = 0.0
    for label, frac in SPLITS:
        acc += frac
        if h < acc:
            return label
    return SPLITS[-1][0]


def loop_pool(manifest: str) -> list[dict]:
    out = []
    with open(manifest) as fh:
        fh.readline()
        for line in fh:
            r = json.loads(line)
            if (
                r["kind"] == "loop"
                and r.get("bpm")
                and r.get("readable")
                and not r.get("discard")
                and "Drums" in set(r.get("families") or ())
            ):
                out.append(r)
    out.sort(key=lambda r: r["file_id"])
    return out


def take(row: dict, rng: np.random.Generator, notch: int):
    """-> (features, beat hit per block, sub-block offset) or None."""
    try:
        x, sr = sf.read(row["path"], dtype="float32", always_2d=True)
    except Exception:
        try:
            x, sr, _ = augment.able.read(row["path"])
        except Exception:
            return None
    if x.size == 0:
        return None
    m = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        m = soxr.resample(m, sr, SAMPLE_RATE, quality="VHQ").astype(np.float32)
    if m.size < SAMPLE_RATE // 2:
        return None

    off = int(rng.integers(m.size))
    need = int(TAKE_S * SAMPLE_RATE)
    tiled = np.resize(np.roll(m, -off), need)

    beat = 60.0 / row["bpm"] * SAMPLE_RATE
    # The grid anchor is measured from the clean file, not assumed to be sample 0.
    phase0 = (row.get("grid_phase", 0.0) - off) % beat

    wet = augment.apply_ir(tiled)
    dbfs = float(rng.uniform(*render.LEVEL_DBFS))
    y = augment.finish(wet, notch, rng, dbfs=dbfs)
    X = features.featurise((y * 32767.0).astype(np.int16), notch, comb=False, hicut=True)

    hit = np.zeros((len(X), 1), dtype=np.float32)
    frac = np.zeros((len(X), 1), dtype=np.float32)
    pos = phase0
    while pos < len(X) * BLOCK_SAMPLES:
        b = int(pos // BLOCK_SAMPLES)
        if b < len(X):
            hit[b, 0] = 1.0
            frac[b, 0] = (pos % BLOCK_SAMPLES) / BLOCK_SAMPLES
        pos += beat
    return X, hit, frac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", type=Path, default=Path("data/grid"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--notch", type=int, default=None, help="default: cycle all three")
    ap.add_argument("--refresh-phase", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pool = verified_loops(a.manifest)
    print(f"{len(pool)} loops with a verified grid phase")
    if a.limit:
        pool = pool[: a.limit]
    acc = {s: {"X": [], "y": [], "off": [], "take": []} for s, _ in SPLITS}
    skip = int(SETTLE_S * SAMPLE_RATE / BLOCK_SAMPLES)
    n, t0 = 0, time.time()

    for i, row in enumerate(pool):
        rng = np.random.default_rng(row["file_id"])
        notch = a.notch or features.NOTCH_HZ[i % len(features.NOTCH_HZ)]
        r = take(row, rng, notch)
        if r is None:
            continue
        X, hit, frac = r
        X, hit, frac = X[skip:], hit[skip:], frac[skip:]
        if len(X) < 200:
            continue
        d = acc[split_of(row)]
        d["X"].append(X)
        d["y"].append(hit)
        d["off"].append(frac)
        d["take"].append(np.full(len(X), n, dtype=np.int32))
        n += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pool)}  {n} takes  {time.time()-t0:.0f}s", flush=True)

    meta = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "real drum loops, beat labels from verified tempo",
        "classes": ["beat"],
        "takes": n,
        "take_seconds": TAKE_S,
        "settle_s": SETTLE_S,
        **features.spec(comb=False, hicut=True),
    }
    for s, _ in SPLITS:
        d = acc[s]
        if not d["X"]:
            continue
        arrs = {k: np.concatenate(v) for k, v in d.items()}
        np.savez(a.out / f"{s}.npz", **arrs)
        meta.setdefault("splits", {})[s] = {
            "blocks": int(len(arrs["X"])),
            "takes": int(len(np.unique(arrs["take"]))),
            "beats": int(arrs["y"].sum()),
        }
        print(f"{s:6} {len(arrs['X']):9,} blocks  {int(arrs['y'].sum()):7,} beats")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta.get("splits", {}), indent=2))


if __name__ == "__main__":
    main()


# --- grid phase verification -------------------------------------------------
# A duration-verified tempo says the loop is a whole number of bars long. It does
# NOT say the loop starts on a beat, and measurement says only about a third do
# within 2% of a beat. Using the loop start as the grid anchor regardless would
# put ~40% of labels on the wrong phase — and would equally corrupt any phase
# metric measured against it. So each loop is checked against its own audio.

PHASE_CACHE = Path("cache/grid_phase.json")
HOP = 48  # 1 ms onset-strength hop


def _onset_strength(m: np.ndarray, hop: int = HOP) -> np.ndarray:
    env = np.sqrt(np.add.reduceat(m**2, np.arange(0, len(m) - hop, hop)) / hop)
    return np.maximum(np.diff(env), 0.0)


def grid_phase(row: dict) -> tuple[float, float] | None:
    """-> (phase in samples, confidence) for this loop's beat grid.

    The period is exact — it comes from the duration-verified tempo. Only the
    phase is estimated, and it is estimated from the clean, full-bandwidth file,
    while the model will see an IR-convolved, whine-injected, band-limited, AGC'd
    version. The label is therefore from a much better-informed view of the same
    audio, not from the model's own kind of guess.

    Confidence is the comb peak over its mean across phases: low values mean the
    loop has no clear grid to anchor to.
    """
    try:
        x, sr = sf.read(row["path"], dtype="float32", always_2d=True)
    except Exception:
        try:
            x, sr, _ = augment.able.read(row["path"])
        except Exception:
            return None
    if x.size == 0:
        return None
    m = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        m = soxr.resample(m, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    od = _onset_strength(m)
    beat_hops = 60.0 / row["bpm"] * SAMPLE_RATE / HOP
    if len(od) < beat_hops * 3:
        return None
    phases = np.arange(0.0, beat_hops, 0.25)
    grid = phases[:, None] + np.arange(int(len(od) / beat_hops))[None, :] * beat_hops
    e = np.where(grid < len(od) - 1,
                 np.interp(np.clip(grid, 0, len(od) - 1), np.arange(len(od)), od),
                 0.0).sum(axis=1)
    if e.max() <= 0:
        return None
    return float(phases[int(np.argmax(e))] * HOP), float(e.max() / (e.mean() + 1e-12))


def phase_table(manifest: str, refresh: bool = False) -> dict[int, tuple[float, float]]:
    cache = {}
    if PHASE_CACHE.exists() and not refresh:
        try:
            cache = json.loads(PHASE_CACHE.read_text())
        except json.JSONDecodeError:
            cache = {}
    pool = loop_pool(manifest)
    dirty = False
    for i, row in enumerate(pool):
        k = str(row["file_id"])
        if k not in cache or not isinstance(cache[k], dict):
            g = grid_phase(row)
            cache[k] = None if g is None else {"phase": g[0], "conf": g[1]}
            dirty = True
            if dirty and i % 500 == 0:
                PHASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
                PHASE_CACHE.write_text(json.dumps(cache))
    if dirty:
        PHASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PHASE_CACHE.write_text(json.dumps(cache))
    return {int(k): (v["phase"], v["conf"]) for k, v in cache.items() if v}


MIN_CONF = 1.6  # comb peak / mean across phases; below this there is no clear grid


def verified_loops(manifest: str, min_conf: float = MIN_CONF) -> list[dict]:
    table = phase_table(manifest)
    out = []
    for row in loop_pool(manifest):
        g = table.get(row["file_id"])
        if g and g[1] >= min_conf:
            r = dict(row)
            r["grid_phase"] = g[0]
            r["grid_conf"] = g[1]
            out.append(r)
    return out
