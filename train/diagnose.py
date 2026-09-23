"""Trace false positives back to the material that caused them.

Re-renders test clips with the same seeds the corpus was built with, so every
detection can be attributed: a percussion note the model mistook for a kick, or
the backing bed. Prints the source files, so they can be auditioned.

    python -m train.diagnose --run runs/melodic-bed --class kick
"""

import argparse
import collections
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from dataset import alc, augment, features, render
from dataset.ableton import DETECTION_CLASSES, Library

from . import data, evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/melodic-bed"))
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--class", dest="cls", default="kick", choices=DETECTION_CLASSES)
    ap.add_argument("--clips", type=int, default=120)
    ap.add_argument("--notch", type=int, default=480)
    a = ap.parse_args()

    from .run import predict  # noqa: circular-free at call time

    m = keras.models.load_model(a.run / "model.keras", compile=False)
    nz = np.load(a.run / "norm.npz")
    mean, scale = nz["mean"], nz["scale"]
    th = json.loads((a.run / "result.json").read_text())["thresholds"][a.cls]
    ci = DETECTION_CLASSES.index(a.cls)

    lib = Library()
    pools = render.pools_from_manifest(a.features.parent / "manifest.jsonl")
    beds = render.backing_pool(a.features.parent / "manifest.jsonl")
    meta = data.meta(a.features)

    from .run import BASELINE_FEATURE  # noqa: F401  (keeps import surface obvious)

    blamed_pad = collections.Counter()
    blamed_file = collections.Counter()
    blamed_bed = collections.Counter()
    unexplained = 0
    n_fp = n_pred = 0

    for fid in alc.drum_clips(lib)[: a.clips]:
        clip = alc.parse(lib.path_of(fid), lib.name_of(fid))
        if clip is None or not clip.notes:
            continue
        from .run import __name__ as _  # noqa
        if _split_of(clip.name) != "test":
            continue
        kit = render.Kit(pools, seed=clip.name)
        bed = None
        cands = beds.get(int(round(clip.tempo)), [])
        if cands:
            row = cands[int(np.random.default_rng(fid).integers(len(cands)))]
            loaded = render._load(row["path"])
            if loaded is not None and loaded.size:
                bed = (loaded, row["name"])
        rng = np.random.default_rng(fid)
        r = render.render(clip, kit, classes=DETECTION_CLASSES, backing=bed,
                          min_seconds=render.AGC_SETTLE_S + render.KEEP_S)
        wet = augment.apply_ir(r.audio)
        dbfs = float(rng.uniform(*render.LEVEL_DBFS))
        y = augment.finish(wet, a.notch, rng, dbfs=dbfs)
        X = features.featurise((y * 32767.0).astype(np.int16))
        skip = int(render.AGC_SETTLE_S * 48000 / 512)

        pred = m.predict(_windows(X, mean, scale), verbose=0)
        prob = np.zeros(len(X)); off = np.zeros(len(X))
        lo = data.PAST
        prob[lo : lo + len(pred["hit"])] = pred["hit"][:, ci]
        off[lo : lo + len(pred["offset"])] = pred["offset"][:, ci]
        prob[:skip] = 0.0

        picks = evaluate.pick_peaks(prob, th)
        pred_ms = (picks + off[picks]) * evaluate.BLOCK_MS
        true_ms = np.array([clip.time_of(n) * 1000 + rep * clip.duration_s * 1000
                            for rep in range(20) for n in clip.notes
                            if n.cls == a.cls]) if clip.notes else np.zeros(0)
        true_ms = true_ms[true_ms < len(X) * evaluate.BLOCK_MS]
        # every non-target note, with what voiced it
        others = [(clip.time_of(n) * 1000 + rep * clip.duration_s * 1000, n)
                  for rep in range(20) for n in clip.notes if n.cls != a.cls]
        others = [(t, n) for t, n in others if t < len(X) * evaluate.BLOCK_MS]

        n_pred += len(picks)
        for p in pred_ms:
            if true_ms.size and np.abs(true_ms - p).min() <= evaluate.TOLERANCE_MS:
                continue
            n_fp += 1
            near = [(abs(t - p), n) for t, n in others if abs(t - p) <= evaluate.TOLERANCE_MS]
            if near:
                _, n = min(near, key=lambda z: z[0])
                blamed_pad[f"{n.cls}: {n.pad}"] += 1
                f = kit.chosen.get((n.cls, n.midi))
                if f:
                    blamed_file[f] += 1
            elif bed is not None:
                blamed_bed[bed[1]] += 1
            else:
                unexplained += 1

    print(f"class {a.cls}: {n_fp} false positives of {n_pred} predictions\n")
    print("blamed on a percussion/other note in the clip:")
    for k, v in blamed_pad.most_common(12):
        print(f"  {v:5}  {k}")
    print("\n  ... voiced by these sample files (audition these):")
    for k, v in blamed_file.most_common(15):
        print(f"  {v:5}  {k}")
    print("\nblamed on the backing bed (no drum tag — audition to confirm):")
    for k, v in blamed_bed.most_common(12):
        print(f"  {v:5}  {k}")
    print(f"\nunexplained (no nearby note, no bed): {unexplained}")


def _windows(X, mean, scale, past=data.PAST, future=data.FUTURE):
    c = np.arange(past, len(X) - future)
    w = X[c[:, None] + np.arange(-past, future + 1)]
    return ((w - mean) / scale).reshape(len(c), -1)


def _split_of(name: str) -> str:
    from dataset.build import split_of
    return split_of(name)


if __name__ == "__main__":
    main()
