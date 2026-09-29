"""Streaming stage 2: the batch estimator (train/tempo.py) run as a tracker.

The published numbers fit one grid per window with no state. A device runs
continuously, so this re-fits every UPDATE blocks on a trailing window and
either takes each new fit (stateless) or holds the current grid until a new fit
explains the recent audio clearly better, or the held grid stops explaining it
(hold). Holding is allowed to latch; it has to let go at a track change.

Scored on playlists (dataset.playlist): usable share of updates, and per
transition the time until the grid is usable again and how long it stays on the
previous song's tempo.

    python -m train.stream runs/n_song_s0 --corpus data/playlists_mel
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import evaluate, gridmetrics, tempo

UPDATE = 47       # 0.5 s
WINDOW = 750      # 8 s, the rendered-clip measurement window
RECENT = 375      # 4 s the held and candidate grids are compared on
MIN_BLOCKS = 188  # 2 s before the first fit
MARGIN = 0.02     # candidate contrast must beat the held grid's by this
FLOOR = 0.03      # a held grid whose contrast falls below this is dropped
SAME_LAG = 0.015
SAME_PHASE = 1.5  # blocks


@dataclass
class Grid:
    lag: float     # blocks
    anchor: float  # a beat position, blocks from the take start


def contrast(act: np.ndarray, g: Grid, t0: int, t1: int) -> float:
    """Mean activation on the grid's beats in [t0, t1) over the mean there."""
    k = np.arange(np.ceil((t0 - g.anchor) / g.lag), np.ceil((t1 - 1 - g.anchor) / g.lag))
    pos = g.anchor + k * g.lag
    if len(pos) == 0:
        return 0.0
    seg = act[t0:t1]
    return float(np.interp(pos - t0, np.arange(len(seg)), seg).mean() - seg.mean())


def _fit(act, t, window):
    w0 = max(0, t - window)
    est = tempo.estimate(act[w0:t])
    return Grid(est.lag, w0 + est.phase_ms / evaluate.BLOCK_MS) if np.isfinite(est.lag) else None


def track(act: np.ndarray, hold: bool, window: int = WINDOW, short: int | None = None):
    """-> list of (block, Grid or None) at every update.

    With `short`, a second fit on the last `short` blocks also competes: the long
    window's grid is the steadier one to hold, the short one sees a new track sooner.
    """
    out, cur = [], None
    for t in range(MIN_BLOCKS, len(act) + 1, UPDATE):
        r0 = max(0, t - RECENT)
        cands = [c for c in (_fit(act, t, window), _fit(act, t, short) if short else None) if c]
        cand = max(cands, key=lambda c: contrast(act, c, r0, t)) if len(cands) > 1 else \
            (cands[0] if cands else None)
        if not hold or cur is None:
            cur = cands[0] if cands else None
            if short and cands and not hold:
                cur = cand
        elif cand is not None:
            c_cur, c_new = contrast(act, cur, r0, t), contrast(act, cand, r0, t)
            d = (cand.anchor - cur.anchor) % cur.lag
            same = (abs(cand.lag / cur.lag - 1) < SAME_LAG
                    and min(d, cur.lag - d) < SAME_PHASE)
            if same:
                cur = cands[0]  # refine from the long window
            elif c_new > c_cur + MARGIN or c_cur < FLOOR:
                cur = cand
        out.append((t, cur))
    return out


TRUTH_WINDOW = 375  # labels must be unmasked over the 4 s before an update to score it


def truth_at(s_take, t):
    """-> (true bpm, a true beat time ms) from the labels before block t, or None."""
    beat, off, period = s_take
    lo = max(0, t - TRUTH_WINDOW)
    if (beat[lo:t] < 0).any() or period[t - 1] <= 0:
        return None
    idx = np.flatnonzero(beat[lo:t] > 0)
    if len(idx) < 2:
        return None
    b = lo + idx[-1]
    return 60.0 * tempo.BLOCK_HZ / float(period[t - 1]), (b + off[b]) * evaluate.BLOCK_MS


def usable(g: Grid | None, truth) -> tuple[bool, bool]:
    """-> (usable, tempo right up to an octave)."""
    if g is None or truth is None:
        return False, False
    bpm = tempo.bpm_of_lag(g.lag)
    ok = tempo.tempo_ok(bpm, truth[0])[1]
    beat_ms = 60000.0 / truth[0]
    d = (g.anchor * evaluate.BLOCK_MS - truth[1]) % beat_ms
    return ok and min(d, beat_ms - d) <= gridmetrics.PHASE_OK_MS, ok


def score(act, s, playlists=None, hold=True, window=WINDOW, short=None) -> dict:
    """Usable share over labelled updates; per transition, re-lock and stale time."""
    from . import seq
    use, relock, stale = [], [], []
    takes = seq.takes(s)
    for ti, g in enumerate(takes):
        a = act[g]
        lab = (s.beat[g][:, 0], s.beat_off[g][:, 0], s.period[g][:, 0])
        res = track(a, hold=hold, window=window, short=short)
        rows = [(t, gr, truth_at(lab, t)) for t, gr in res]
        use += [usable(gr, tr)[0] for t, gr, tr in rows if tr is not None]
        if playlists is None:
            continue
        info = playlists[ti]
        segs = info["segments"]
        for j, tr_ in enumerate(info["transitions"]):
            nxt, prev = segs[j + 1], segs[j]
            if nxt["status"] != "confirmed":
                continue
            start = max(tr_["block"], 0)
            if tr_["kind"] == "fade":
                start = segs[j]["end_block"]
            end = nxt["end_block"]
            after = [(t, gr, tru) for t, gr, tru in rows if start < t <= end and tru is not None]
            if not after:
                continue
            t_ok = next((t for i, (t, gr, tru) in enumerate(after)
                         if all(usable(g2, tr2)[0] for _, g2, tr2 in after[i:i + 3])), None)
            relock.append((t_ok - start) / tempo.BLOCK_HZ if t_ok is not None else np.inf)
            if prev["bpm"] and not tempo.tempo_ok(prev["bpm"], nxt["bpm"])[1]:
                old = [gr is not None and tempo.tempo_ok(tempo.bpm_of_lag(gr.lag), prev["bpm"])[1]
                       for t, gr, _ in after if t - start <= 10 * tempo.BLOCK_HZ]
                stale.append(np.mean(old) * 10.0 if old else 0.0)
    out = {"updates": len(use), "usable": float(np.mean(use)) if use else None}
    if relock:
        r = np.array(relock)
        out.update(transitions=len(r), relock_median_s=float(np.median(r)),
                   relock_5s=float(np.mean(r <= 5)), relock_10s=float(np.mean(r <= 10)),
                   never=float(np.mean(~np.isfinite(r))))
    if stale:
        out["stale_s_first10"] = float(np.mean(stale))
    return out


VARIANTS = {"batch8": (False, 750, None), "batch4": (False, 375, None),
            "hold8": (True, 750, None), "hold4": (True, 375, None),
            "hold8+4": (True, 750, 375), "hold8+2": (True, 750, 188)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--corpus", default="data/playlists_mel")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from . import gpu  # noqa: F401  must precede keras
    import keras

    from . import data, scoreboard
    s_raw = data.load_many([Path(a.corpus)], a.split, mel="raw")
    pj = Path(a.corpus) / "playlists.json"
    playlists = json.loads(pj.read_text())[a.split] if pj.exists() else None
    res = json.load(open(a.out)) if a.out and Path(a.out).exists() else {}
    for run in a.runs:
        r = Path(run)
        info = json.loads((r / "result.json").read_text())
        m = keras.models.load_model(r / "model.keras", compile=False)
        nz = np.load(r / "norm.npz")
        s = data.for_run(s_raw, a.corpus, a.split, info)
        act = scoreboard.run_activation(r, m, s, nz["mean"], nz["scale"], info)
        for v in a.variants:
            hold, window, short = VARIANTS[v]
            row = score(act, s, playlists, hold=hold, window=window, short=short)
            res[f"{run}:{Path(a.corpus).name}:{a.split}:{v}"] = row
            print(f"{run:24} {v:7} " + "  ".join(
                f"{k} {x:.3f}" if isinstance(x, float) else f"{k} {x}" for k, x in row.items()),
                flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
