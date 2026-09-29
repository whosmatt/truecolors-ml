"""Usable grid on hand-labelled songs, per 12 s window of included audio.

Each window gets its own stage-2 estimate from the model's activation and is
scored against the confirmed song grid with gridmetrics' rule (tempo right up
to an octave, phase within 25 ms). Windows touching an excluded span, or inside
the lookback at the song start, are skipped.

    python -m train.songeval runs/m_cand_s0 runs/n_song_s0 --out x.json
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, evaluate, gridmetrics, scoreboard, seq, tempo

WINDOW = 1125  # 12 s of blocks
START = 461    # the equal-window start used by train.compare


def windows(act, s):
    out = []
    for g in seq.takes(s):
        beat, off, period = s.beat[g][:, 0], s.beat_off[g][:, 0], s.period[g][:, 0]
        if (period <= 0).all():
            continue
        a = act[g]
        for w0 in range(START, len(g) - WINDOW + 1, WINDOW):
            w = slice(w0, w0 + WINDOW)
            if (beat[w] < 0).any():
                continue
            idx = np.flatnonzero(beat[w] > 0)
            if len(idx) < 8:
                continue
            true_bpm = 60.0 * tempo.BLOCK_HZ / float(period[w][0])
            beat_ms = 60000.0 / true_bpm
            phase = float(((idx[0] + off[w][idx[0]]) * evaluate.BLOCK_MS) % beat_ms)
            est = tempo.estimate(a[w])
            e1, e2 = tempo.tempo_ok(est.bpm, true_bpm)
            p = tempo.grid_error_ms(est, true_bpm, phase) if e2 else float("nan")
            out.append((e1, e2, p))
    return np.array(out, dtype=float).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--corpus", default="data/songs")
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sets = {sp: data.load_many([Path(a.corpus)], sp, mel="raw") for sp in a.splits}
    res = json.load(open(a.out)) if a.out and Path(a.out).exists() else {}
    for run in a.runs:
        r = Path(run)
        m = keras.models.load_model(r / "model.keras", compile=False)
        nz = np.load(r / "norm.npz")
        spec = scoreboard.run_spec(r)
        info = json.loads((r / "result.json").read_text())
        row = {}
        for sp, s in sets.items():
            s = data.for_run(s, a.corpus, sp, info)
            act = scoreboard.run_activation(r, m, s, nz["mean"], nz["scale"], info)
            w = windows(act, s)
            ok = w[:, 1] > 0
            ph = w[ok, 2]
            row[sp] = {"songs": int(len(seq.takes(s))), "windows": int(len(w)),
                       "usable": float(np.mean(ok & (w[:, 2] <= gridmetrics.PHASE_OK_MS))),
                       "tempo": float(np.mean(w[:, 0])), "octave": float(np.mean(ok)),
                       "phase_median_ms": float(np.median(ph)) if len(ph) else None}
        res[run] = row
        print(run, json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
