"""Music gate, scored per take: the firmware gates on a smoothed output, so
the question is whether a whole take lands on the right side of a threshold.

Groups: rendered clips, real drum loops (never labelled for the v1 head),
non-music noise takes, and silent takes (median level at the floor).

    python -m train.musiceval runs/e_music runs/f_music2
"""

import argparse
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, mres

THRESHOLDS = (0.5, 0.7, 0.8)
SILENT_DB = 55.0  # take median spl; the front-end floor is 50 dB
PER_TAKE = 24


class Int8:
    """The exported model's music output, matched by signature position."""

    def __init__(self, path: Path, heads):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        outs = {int(d["name"].rsplit(":", 1)[1]): d for d in self.it.get_output_details()}
        self.out = outs[heads.index("music")]

    def predict(self, x, **_):
        s, z = self.inp["quantization"]
        qs, qz = self.out["quantization"]
        q = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        r = np.empty((len(x), 1), dtype=np.float32)
        for i, row in enumerate(q):
            self.it.set_tensor(self.inp["index"], row[None])
            self.it.invoke()
            r[i] = (self.it.get_tensor(self.out["index"])[0].astype(np.float32) - qz) * qs
        return {"music": r}


def take_medians(model, s, mean, scale, spec, rng):
    c = mres.valid_centres(s.take, spec)
    ids = s.take[c]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    picks = np.concatenate([rng.choice(g, min(PER_TAKE, len(g)), replace=False)
                            for g in np.split(c, starts[1:])])
    w = mres.gather(s.X, mres.cumsum(s.X), picks, spec).reshape(len(picks), spec.frames, -1)
    x = ((w - mean) / scale).reshape(len(picks), -1).astype(np.float32)
    p = np.asarray(model.predict(x, verbose=0, batch_size=8192)["music"])[:, 0]
    t = s.take[picks]
    u = np.unique(t)
    med = np.array([np.median(p[t == k]) for k in u])
    lvl = np.array([np.median(s.X[s.take == k, 11]) for k in u])
    return med, lvl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clips", type=Path, default=Path("data/features3"))
    ap.add_argument("--loops", type=Path, default=Path("data/loops"))
    ap.add_argument("--noise", type=Path, default=Path("data/noise"))
    ap.add_argument("--tflite", type=Path, nargs="*", default=[],
                    help="one exported model per run, scored instead of the float model")
    a = ap.parse_args()
    spec = mres.Spec()
    corpora = {n: data.load(d, a.split) for n, d in
               (("clips", a.clips), ("loops", a.loops), ("noise", a.noise))}
    for k, run in enumerate(a.runs):
        nz = np.load(run / "norm.npz")
        m = keras.models.load_model(run / "model.keras", compile=False)
        if a.tflite:
            heads = list(m.output_names) if hasattr(m, "output_names") else list(m.output.keys())
            m = Int8(a.tflite[k], heads)
        rng = np.random.default_rng(0)
        groups = {}
        for n, s in corpora.items():
            med, lvl = take_medians(m, s, nz["mean"], nz["scale"], spec, rng)
            if n == "noise":
                groups["non-music"] = (med[lvl >= SILENT_DB], False)
                groups["silence"] = (med[lvl < SILENT_DB], False)
            else:
                groups["rendered" if n == "clips" else "real loops"] = (med, True)
        print(f"{run}{' int8' if a.tflite else ''}  (per-take median, {a.split})")
        print(f"  {'group':12} {'takes':>6} {'median':>7}  " +
              "  ".join(f"@{t}" for t in THRESHOLDS) + "   (% on the correct side)")
        for g, (med, pos) in groups.items():
            ok = [np.mean(med >= t if pos else med < t) * 100 for t in THRESHOLDS]
            print(f"  {g:12} {len(med):6} {np.median(med):7.2f}  " +
                  "  ".join(f"{v:4.1f}" for v in ok))


if __name__ == "__main__":
    main()
