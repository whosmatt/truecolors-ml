"""Inference-time context ablation: blank (0 = the training mean) or shuffle
(another window's values) one span of frames, and score the grid. Measures what
a trained model relies on, not what it needs: retraining without a tier is the
other half.

    python -m train.ablate runs/i_ctrl_s0 runs/i_ctrl_s1 --out x.json
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, gridmetrics, mres


def spans(spec: mres.Spec) -> dict:
    fine = spec.fine_past + spec.fine_future + 1
    mid = fine + spec.mid_frames
    return {"fine": (0, fine), "fine, oldest 8": (0, 8), "mid": (fine, mid),
            "coarse": (mid, spec.frames), "coarse, oldest 6": (mid + 6, spec.frames),
            "mid + coarse": (fine, spec.frames)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--clips", type=Path, default=Path("data/features3"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    spec = mres.Spec()
    te = data.load_many([a.clips], "test")
    res = {}
    for run in a.runs:
        m = keras.models.load_model(Path(run) / "model.keras", compile=False)
        nz = np.load(Path(run) / "norm.npz")
        ds = mres.MResWindows(te, nz["mean"], nz["scale"], spec, targets=())
        X = np.concatenate([ds.window(ds.order[i:i + 65536])
                            for i in range(0, len(ds.order), 65536)]).astype(np.float32)
        X = X.reshape(len(X), spec.frames, -1)
        rng = np.random.default_rng(0)

        def score(Xm):
            out = m.predict(Xm.reshape(len(Xm), -1), verbose=0, batch_size=16384)
            act = np.zeros(len(te.X), np.float32)
            act[ds.order] = out["beat"][:, 0]
            return gridmetrics.per_take(act, te)

        r = {"none": score(X)}
        for name, (lo, hi) in spans(spec).items():
            Xz = X.copy()
            Xz[:, lo:hi] = 0.0
            r[f"{name}: blank"] = score(Xz)
            Xs = X.copy()
            Xs[:, lo:hi] = X[rng.permutation(len(X)), lo:hi]
            r[f"{name}: shuffle"] = score(Xs)
        res[run] = r
        for k, v in r.items():
            print(f"{run} {k:26} usable {v['usable']:.3f} octave {v['octave']:.3f}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
