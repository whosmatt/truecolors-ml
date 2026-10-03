"""Non-music takes: the negative class for "is anything musical playing".

Every take in the rendered and loop corpora contains drums, so the model has no
way to represent "someone is talking in a quiet room" — which is exactly what it
tracks a tempo on when deployed.

Sources: speech, field recordings, atmospheres and sound effects from the
library, plus takes of nothing at all (the measured room floor and coil whine),
because the most common non-music state is a quiet room with the laser on.

Solo Voice is deliberately excluded — a cappella singing is music by any
definition the product cares about.

    python -m dataset.noiseset --out data/noise --limit 3000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

from . import augment, features, gridset, render
from .ableton import AUDIO_TYPES, KEY_CKEY, KEY_KEYW, Library

MASK = -1.0
TAKE_S = 16.0
SETTLE_S = 4.0
NON_MUSIC_TAGS = (
    "Sounds|Voice|Speech",
    "Sounds|Ambience & FX|Field & Foley",
    "Sounds|Ambience & FX|Atmosphere",
    "Sounds|Ambience & FX|Sound FX",
)
SILENT_FRACTION = 0.15  # takes that are only room tone and coil whine
# Real device captures (miccap, 2026-10-03): typing and room activity, fridge,
# laser-off baselines, coil whine sweeps with the fan. Raw samples, exactly what
# the device sees. Each capture's last 10 s is held out as one contiguous val
# take, so neighbouring windows never straddle the split.
DEVICE_CAPTURES = (
    "data/ambient/2026-10-03/ambient_*.raw",
    "data/whine/2026-10-03/quiet_*.raw",
    "data/whine/2026-10-03/sweep*.raw",
    "data/whine/2026-10-03/rainbow*.raw",
    "data/ir/2026-10-03/dev_quiet.raw",
)
DEVICE_VAL_S = 10.0
# Train repeats: ~2% of the noise blocks as recorded. Room activity also comes
# louder or closer; self-noise (baselines, whine) only at its real level.
DEVICE_TRAIN_GAINS_DB = {"ambient": (0.0, 6.0, 12.0)}
DEVICE_TRAIN_REPEATS = 3


def device_takes():
    """-> [(name, split, notch, int16 samples)] from the raw device captures."""
    import glob
    out = []
    for pat in DEVICE_CAPTURES:
        for path in sorted(glob.glob(pat)):
            x = np.fromfile(path, dtype="<i2")
            name = Path(path).stem
            hz = next((h for h in features.NOTCH_HZ if name.startswith(("sweep%d" % h, "rainbow%d" % h))), 0)
            cut = len(x) - int(DEVICE_VAL_S * SAMPLE_RATE)
            gains = DEVICE_TRAIN_GAINS_DB.get(name.split("_")[0], (0.0,) * DEVICE_TRAIN_REPEATS)
            for g in gains:
                tr = x[:cut]
                if g:
                    tr = np.clip(np.round(tr.astype(np.float64) * 10 ** (g / 20)), -32768, 32767).astype("<i2")
                out.append((f"{name}{'+%gdB' % g if g else ''}", "train", hz, tr))
            out.append((name, "val", hz, x[cut - int(SETTLE_S * SAMPLE_RATE):]))
    return out


def pool(lib: Library, manifest: str) -> list[dict]:
    ok = {}
    with open(manifest) as fh:
        fh.readline()
        for line in fh:
            r = json.loads(line)
            if r.get("readable") and r.get("duration_s", 0) >= 4.0:
                ok[r["file_id"]] = r
    ids = set()
    for tag in NON_MUSIC_TAGS:
        ids |= {
            r[0]
            for r in lib._q(
                f"""select f.file_id from files f
                    join metadata m on m.file_id = f.file_id
                    join metadata_values v on v.id = m.value_id
                    where m.key in ({KEY_KEYW}, {KEY_CKEY}) and v.value = ?
                      and f.file_type in {AUDIO_TYPES}""",
                (tag,),
            )
        }
    return [ok[i] for i in sorted(ids) if i in ok]


def take(row: dict | None, rng: np.random.Generator, notch: int):
    """A non-music take. `row` None means silence: room tone and whine only."""
    need = int(TAKE_S * SAMPLE_RATE)
    if row is None:
        wet = np.zeros(need, dtype=np.float32)
    else:
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
        if m.size < SAMPLE_RATE:
            return None
        tiled = np.resize(np.roll(m, -int(rng.integers(m.size))), need)
        wet = augment.apply_ir(tiled, rng)
    dbfs = float(rng.uniform(*render.LEVEL_DBFS))
    y = augment.finish(wet, notch, rng, dbfs=dbfs)
    return features.featurise((y * 32767.0).astype(np.int16), hicut=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", type=Path, default=Path("data/noise"))
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    lib = Library()
    rows = pool(lib, a.manifest)[: a.limit]
    n_silent = int(len(rows) * SILENT_FRACTION)
    items = rows + [None] * n_silent
    skip = int(SETTLE_S * SAMPLE_RATE / BLOCK_SAMPLES)
    acc = {s: {k: [] for k in ("X", "y", "off", "beat", "beat_off", "downbeat",
                               "downbeat_off", "phrase", "period", "music",
                               "take", "notch")} for s, _ in gridset.SPLITS}
    n, t0 = 0, time.time()

    for i, row in enumerate(items):
        seed = (row["file_id"] if row else 10**9 + i)
        rng = np.random.default_rng(seed)
        notch = features.NOTCH_CYCLE[i % len(features.NOTCH_CYCLE)]
        X = take(row, rng, notch)
        if X is None or len(X) <= skip + 300:
            continue
        X = X[skip:]
        m = len(X)
        split = gridset.split_of(row) if row else ("train" if i % 10 else "val")
        d = acc[split]
        d["X"].append(X)
        for k, w in (("y", 4), ("off", 3), ("beat", 1), ("beat_off", 1),
                     ("downbeat", 1), ("downbeat_off", 1), ("phrase", 1),
                     ("period", 1)):
            d[k].append(np.full((m, w), MASK, dtype=np.float32))
        d["music"].append(np.zeros((m, 1), dtype=np.float32))
        d["take"].append(np.full(m, n, dtype=np.int32))
        d["notch"].append(np.full(m, notch, dtype=np.int16))
        n += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(items)}  {n} takes  {time.time()-t0:.0f}s", flush=True)

    dev = []
    for name, split, hz, x in device_takes():
        X = features.featurise(x, hicut=True)[skip:]
        m = len(X)
        d = acc[split]
        d["X"].append(X)
        for k, w in (("y", 4), ("off", 3), ("beat", 1), ("beat_off", 1),
                     ("downbeat", 1), ("downbeat_off", 1), ("phrase", 1),
                     ("period", 1)):
            d[k].append(np.full((m, w), MASK, dtype=np.float32))
        d["music"].append(np.zeros((m, 1), dtype=np.float32))
        d["take"].append(np.full(m, n, dtype=np.int32))
        d["notch"].append(np.full(m, hz, dtype=np.int16))
        dev.append(f"{name}:{split}")
        n += 1
    print(f"  device captures: {len(dev)} takes", flush=True)

    meta = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "non-music: speech, foley, atmosphere, sound fx, plus silence, plus raw device captures",
            "device_captures": dev,
            "tags": list(NON_MUSIC_TAGS), "silent_fraction": SILENT_FRACTION,
            "takes": n, **features.spec(hicut=True)}
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
