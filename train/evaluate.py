"""Onset evaluation. Timing error in milliseconds first, F1 second.

A detector that finds every kick but places it 30 ms late reconstructs a useless
grid, so the headline number here is the millisecond error of matched onsets, not
the classification score.
"""

from dataclasses import dataclass, asdict

import numpy as np

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

BLOCK_MS = BLOCK_SAMPLES / SAMPLE_RATE * 1000.0
TOLERANCE_MS = 50.0  # the usual MIR onset window
MIN_SEP_BLOCKS = 3   # ~32 ms; 32nd notes at 180 BPM are 41 ms apart


@dataclass
class Score:
    n_true: int
    n_pred: int
    tp: int
    precision: float
    recall: float
    f1: float
    median_abs_ms: float | None
    p90_abs_ms: float | None
    bias_ms: float | None

    def asdict(self):
        return asdict(self)


def pick_peaks(prob: np.ndarray, threshold: float, min_sep: int = MIN_SEP_BLOCKS) -> np.ndarray:
    """Local maxima above threshold, thinned to one per min_sep blocks."""
    n = len(prob)
    cand = np.flatnonzero(prob >= threshold)
    if cand.size == 0:
        return cand
    keep, last = [], -10**9
    for i in cand[np.argsort(-prob[cand], kind="stable")]:
        if all(abs(i - j) >= min_sep for j in keep):
            keep.append(int(i))
    keep.sort()
    # drop anything that is not a local maximum in its neighbourhood
    out = [i for i in keep if prob[i] >= prob[max(0, i - 1) : min(n, i + 2)].max()]
    return np.asarray(out, dtype=np.int64)


def match(pred_ms: np.ndarray, true_ms: np.ndarray, tol: float = TOLERANCE_MS):
    """Greedy nearest-neighbour matching. -> (errors, tp)."""
    if pred_ms.size == 0 or true_ms.size == 0:
        return np.zeros(0), 0
    used = np.zeros(true_ms.size, dtype=bool)
    errs = []
    order = np.argsort(pred_ms)
    for p in pred_ms[order]:
        d = np.abs(true_ms - p)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            errs.append(p - true_ms[j])
    return np.asarray(errs), int(used.sum())


def score_class(
    prob: np.ndarray,
    offset: np.ndarray | None,
    truth_block: np.ndarray,
    truth_off: np.ndarray,
    takes: np.ndarray,
    threshold: float,
    tol: float = TOLERANCE_MS,
) -> Score:
    """Evaluate one class over one split, take by take."""
    all_err, tp, n_pred, n_true = [], 0, 0, 0
    for t in np.unique(takes):
        m = takes == t
        p = prob[m]
        idx = np.flatnonzero(m)
        picks = pick_peaks(p, threshold)
        # Without an offset head the onset can only be placed at the block centre.
        frac = offset[m][picks] if offset is not None else np.full(len(picks), 0.5)
        pred_ms = (picks + frac) * BLOCK_MS
        tm = truth_block[m]
        ti = np.flatnonzero(tm > 0)
        true_ms = (ti + truth_off[m][ti]) * BLOCK_MS
        e, k = match(pred_ms, true_ms, tol)
        all_err.append(e)
        tp += k
        n_pred += len(picks)
        n_true += len(ti)
    err = np.concatenate(all_err) if all_err else np.zeros(0)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_true if n_true else 0.0
    return Score(
        n_true=n_true,
        n_pred=n_pred,
        tp=tp,
        precision=prec,
        recall=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        median_abs_ms=float(np.median(np.abs(err))) if err.size else None,
        p90_abs_ms=float(np.percentile(np.abs(err), 90)) if err.size else None,
        bias_ms=float(np.median(err)) if err.size else None,
    )


def best_threshold(prob, offset, ty, toff, takes, grid=None) -> tuple[float, Score]:
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.05)
    best, bs = grid[0], None
    for th in grid:
        s = score_class(prob, offset, ty, toff, takes, float(th))
        if bs is None or s.f1 > bs.f1:
            best, bs = float(th), s
    return best, bs


def report(name: str, scores: dict[str, Score]) -> None:
    print(f"\n{name}")
    print(f"  {'class':7} {'median ms':>10} {'p90 ms':>8} {'bias ms':>8} "
          f"{'F1':>6} {'prec':>6} {'rec':>6}  {'n':>6}")
    for cls, s in scores.items():
        med = f"{s.median_abs_ms:10.2f}" if s.median_abs_ms is not None else f"{'-':>10}"
        p90 = f"{s.p90_abs_ms:8.2f}" if s.p90_abs_ms is not None else f"{'-':>8}"
        bias = f"{s.bias_ms:+8.2f}" if s.bias_ms is not None else f"{'-':>8}"
        print(f"  {cls:7} {med} {p90} {bias} {s.f1:6.3f} {s.precision:6.3f} "
              f"{s.recall:6.3f}  {s.n_true:6}")
