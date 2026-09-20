"""Shared end-to-end grid metrics, so every approach is scored identically.

`usable` is the product metric: the tempo is right up to an octave AND the grid
lands on real beats. A half-time grid that hits every other beat is fine for a
metronome; a grid a half-beat off is not, however close its tempo.

Phase is reported split by whether the tempo was exact, because folding phase
error against the true beat makes an octave error look like a phase error even
when the grid is landing perfectly.
"""

import numpy as np

from . import evaluate, tempo

PHASE_OK_MS = 25.0


def per_take(act: np.ndarray, split) -> dict:
    usable = oc = n = 0
    ph_exact, ph_octave, all_ph = [], [], []
    for t in np.unique(split.take):
        m = split.take == t
        idx = np.flatnonzero(split.beat[m][:, 0] > 0)
        if len(idx) < 8:
            continue
        bt = (idx + split.beat_off[m][idx, 0]) * evaluate.BLOCK_MS
        true_bpm = 60000.0 / float(np.median(np.diff(bt)))
        n += 1
        est = tempo.estimate(act[m])
        e1, e2 = tempo.tempo_ok(est.bpm, true_bpm)
        if not e2:
            continue
        oc += 1
        p = tempo.grid_error_ms(est, true_bpm, float(bt[0] % (60000.0 / true_bpm)))
        all_ph.append(p)
        (ph_exact if e1 else ph_octave).append(p)
        if p <= PHASE_OK_MS:
            usable += 1
    f = lambda v: float(np.median(v)) if v else float("nan")
    return {
        "takes": n,
        "usable": usable / max(n, 1),
        "octave": oc / max(n, 1),
        "phase_exact": f(ph_exact),
        "phase_octave": f(ph_octave),
        "phase_p90": float(np.percentile(all_ph, 90)) if all_ph else float("nan"),
    }
