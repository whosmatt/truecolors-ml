"""Do loops start on a beat? Models that never saw loop phase labels (rendered
clips only) estimate each loop's grid; it is compared with the grid anchored at
the file start, as a signed fraction of a beat, on tempo-exact takes only.

    python -m train.loopstart data/loops_grid runs/e_base runs/i_ctrl_s0 --out x.json
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, evaluate, scoreboard, seq, tempo


def offsets(act, s):
    """-> signed fraction of a beat (0 = on the file-start grid) per tempo-exact take."""
    out = []
    for g in seq.takes(s):
        idx = np.flatnonzero(s.beat[g][:, 0] > 0)
        bt = (idx + s.beat_off[g][idx, 0]) * evaluate.BLOCK_MS
        beat = float(np.median(np.diff(bt)))
        est = tempo.estimate(act[g])
        if tempo.tempo_ok(est.bpm, 60000.0 / beat)[0]:
            out.append((((est.phase_ms - bt[0]) / beat) % 1.0, beat))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    s = data.load_many([a.corpus], "test")
    res = json.loads(a.out.read_text()) if a.out and a.out.exists() else {}
    for run in a.runs:
        r = Path(run)
        m = keras.models.load_model(r / "model.keras", compile=False)
        nz = np.load(r / "norm.npz")
        act = scoreboard.activation(m, s, nz["mean"], nz["scale"], scoreboard.run_spec(r))[0]
        o = offsets(act, s)
        f, beat = o[:, 0], o[:, 1]
        ms = np.minimum(f, 1 - f) * beat
        eighths = np.histogram((f + 1 / 16) % 1.0, bins=8, range=(0, 1))[0]
        row = {"takes": int(len(np.unique(s.take))), "tempo_exact": int(len(f)),
               "within_25ms": float(np.mean(ms <= 25.0)), "median_ms": float(np.median(ms)),
               "eighths": eighths.tolist()}
        res[f"{a.corpus.name}:{run}"] = row
        print(f"{a.corpus.name} {run}: {row}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
