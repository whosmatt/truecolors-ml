"""Score any model against the same baseline, on the same test sets.

Every change from here — music detection, downbeat, hi-cut, ablations — has to be
comparable, so scoring lives in one place. Two test sets, because they answer
different questions:

  rendered clips   exact grid, so tempo AND phase are scored
  real drum loops  verified tempo only, but the only non-synthetic material

    python -m train.scoreboard --add runs/data3:baseline --add runs/x:with-music
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, evaluate, gridmetrics, mres, tempo

BOARD = Path("results/scoreboard.json")


def activation(model, split, mean, scale, spec, key="beat") -> np.ndarray:
    """Dense per-block activation, whatever window the model expects."""
    plain = model.input_shape[-1] == (spec.fine_past + spec.fine_future + 1) * 12
    if plain:
        c = np.arange(spec.fine_past, len(split.X) - spec.fine_future)
        w = split.X[c[:, None] + np.arange(-spec.fine_past, spec.fine_future + 1)]
        xb = ((w - mean) / scale).reshape(len(c), -1).astype(np.float32)
        out = model.predict(xb, verbose=0, batch_size=8192)
    else:
        ds = mres.MResWindows(split, mean, scale, spec, targets=(), batch=8192)
        c = ds.order
        out = model.predict(ds, verbose=0)
    if key not in out:
        key = "hit"
    v = out[key]
    a = np.zeros(len(split.X), dtype=np.float32)
    a[c] = v[:, 0] if v.shape[1] == 1 else np.maximum(v[:, 0], v[:, 1])
    return a, out, c


def score_clips(model, te, mean, scale, spec) -> dict:
    a, _, _ = activation(model, te, mean, scale, spec)
    return gridmetrics.per_take(a, te)


def score_loops(model, lo, mean, scale, spec) -> dict:
    a, _, _ = activation(model, lo, mean, scale, spec)
    ex = oc = n = 0
    for t in np.unique(lo.take):
        m = lo.take == t
        true = 60.0 * tempo.BLOCK_HZ / float(np.median(lo.period[m][:, 0]))
        n += 1
        est = tempo.estimate(a[m])
        e1, e2 = tempo.tempo_ok(est.bpm, true)
        ex += e1
        oc += e2
    return {"takes": n, "tempo": ex / max(n, 1), "octave": oc / max(n, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--add", action="append", default=[],
                    help="run_dir:label, repeatable")
    ap.add_argument("--clips", type=Path, default=Path("data/features3"))
    ap.add_argument("--loops", type=Path, default=Path("data/loops"))
    ap.add_argument("--board", type=Path, default=BOARD)
    ap.add_argument("--no-loops", action="store_true")
    a = ap.parse_args()

    spec = mres.Spec()
    te = data.load_many([a.clips], "test")
    lo = None if a.no_loops else data.load_many([a.loops], "test")
    board = json.loads(a.board.read_text()) if a.board.exists() else {}

    for item in a.add:
        run, _, label = item.partition(":")
        label = label or Path(run).name
        model = keras.models.load_model(Path(run) / "model.keras", compile=False)
        nz = np.load(Path(run) / "norm.npz")
        entry = {"run": run, "clips": score_clips(model, te, nz["mean"], nz["scale"], spec)}
        if lo is not None:
            entry["loops"] = score_loops(model, lo, nz["mean"], nz["scale"], spec)
        try:
            entry["macs"] = json.loads((Path(run) / "result.json").read_text())["macs"]
        except Exception:
            entry["macs"] = None
        board[label] = entry
        print(f"scored {label}", flush=True)

    a.board.parent.mkdir(parents=True, exist_ok=True)
    a.board.write_text(json.dumps(board, indent=2))

    print(f"\n{'':22} {'rendered clips':^26}  {'real loops':^18}")
    print(f"{'model':22} {'usable':>8} {'octave':>8} {'phase':>8}  {'tempo':>8} {'octave':>8}  {'MACs':>8}")
    for label, e in board.items():
        c, l = e["clips"], e.get("loops")
        lt = f"{l['tempo']:8.1%} {l['octave']:8.1%}" if l else f"{'-':>8} {'-':>8}"
        print(f"{label:22} {c['usable']:8.1%} {c['octave']:8.1%} {c['phase_exact']:7.2f}ms  "
              f"{lt}  {e['macs'] or 0:8,}")


if __name__ == "__main__":
    main()
