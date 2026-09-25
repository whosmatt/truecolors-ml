"""Score window and sequence runs on identical blocks, on five test sets.

Each take's activation is zeroed before --eval-start, the longest lookback among
the variants compared, so every tempo estimate sees the same audio whatever its
context. Test sets: rendered clips (grid), real drum and melodic loops (tempo),
and the same loops built with --grid (grid anchored at the file start).

    python -m train.compare runs/i_ctrl_s0 runs/j_tcn32_s0 --out runs/compare6.json
"""
import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, gridmetrics, scoreboard, seq, tempo

GRID = ("loops_grid", "melodic_grid")


def pos_in_take(s):
    starts = np.flatnonzero(np.r_[True, s.take[1:] != s.take[:-1]])
    first = np.repeat(starts, np.diff(np.r_[starts, len(s.take)]))
    return np.arange(len(s.take)) - first


def tempo_score(act, s):
    ex = oc = n = 0
    for g in seq.takes(s):
        true = 60.0 * tempo.BLOCK_HZ / float(np.median(s.period[g][:, 0]))
        e1, e2 = tempo.tempo_ok(tempo.estimate(act[g]).bpm, true)
        ex, oc, n = ex + e1, oc + e2, n + 1
    return {"takes": n, "tempo": ex / n, "octave": oc / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--eval-start", type=int, default=461)
    ap.add_argument("--melodic", default="data/melodic")
    ap.add_argument("--clips", default="data/features3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sets = {"clips": data.load_many([Path(a.clips)], "test"),
            "loops": data.load_many([Path("data/loops")], "test")}
    if Path(a.melodic, "test.npz").exists():
        sets["melodic"] = data.load_many([Path(a.melodic)], "test")
    for g in GRID:
        sets[g] = data.load_many([Path("data") / g], "test")

    res = json.load(open(a.out)) if a.out and Path(a.out).exists() else {}
    for run in a.runs:
        r = Path(run)
        m = keras.models.load_model(r / "model.keras", compile=False)
        nz = np.load(r / "norm.npz")
        info = json.loads((r / "result.json").read_text())
        is_seq = info.get("approach") == "6-sequence"
        row = {"macs": info.get("macs")}
        for name, s in sets.items():
            if is_seq:
                act = seq.activation(m, s, nz["mean"], nz["scale"])
            else:
                act = scoreboard.activation(m, s, nz["mean"], nz["scale"], scoreboard.run_spec(r))[0]
            act = np.where(pos_in_take(s) >= a.eval_start, act, 0.0).astype(np.float32)
            row[name] = (gridmetrics.per_take(act, s) if name in ("clips",) + GRID
                         else tempo_score(act, s))
        res[run] = row
        c = row["clips"]
        extra = "  ".join(f"{k} {row[k]['tempo']:.3f}/{row[k]['octave']:.3f}"
                          for k in sets if k not in ("clips",) + GRID)
        extra += "  " + "  ".join(f"{k} usable {row[k]['usable']:.3f} ph {row[k]['phase_exact']:.1f}"
                                  for k in GRID)
        print(f"{run:26} usable {c['usable']:.3f} octave {c['octave']:.3f} "
              f"phase {c['phase_exact']:.2f} | {extra} | MACs {row['macs']}", flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
