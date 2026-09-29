"""BeatNet against the hand-labelled grids, next to our model, on identical terms.

Every method is reduced to beat events on the song's timeline and scored three
ways against the confirmed grid, over included spans only:

    global: one constant-tempo grid fitted through all events (songgrid.fit_grid),
            the auto-labelling question; tempo within 0.1% (and up to an octave),
            phase difference at the song's middle.
    window: a grid fitted per 12 s window, usable if the tempo is right up to an
            octave and the phase within 25 ms (gridmetrics' rule).
    F:      beat F-measure at +-70 ms, the usual beat-tracking score.

Methods: BeatNet offline (DBN) and online (particle filter, causal) from
songs/beatnet/<cond>, and model detections from a songgrid analysis dir.

    python -m train.beatnet_eval --ours songs/auto_cv_d0 --out runs/beatnet_eval.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dataset.songcheck import phase_diff_ms
from dataset.songgrid import SONGS, fit_grid

from . import gridmetrics, tempo

WINDOW_S = 12.0
F_TOL = 0.07


def truth(lab: dict, dur: float):
    T = 60.0 / lab["bpm"]
    k = np.arange(np.ceil(-lab["t0_s"] / T), np.floor((dur - lab["t0_s"]) / T) + 1)
    beats = lab["t0_s"] + k * T
    ex = [(s["start_s"], s["end_s"]) for s in lab.get("sections") or () if not s.get("include", True)]
    return beats, ex


def included(spans_ex, a, b) -> bool:
    return not any(s < b and e > a for s, e in spans_ex)


def pulses(ev: np.ndarray, w: np.ndarray, a: float, b: float) -> np.ndarray:
    """Events in [a, b) as a block-rate pulse train, weighted, for stage 2."""
    n = int((b - a) * tempo.BLOCK_HZ)
    x = np.zeros(n, np.float32)
    m = (ev >= a) & (ev < b)
    np.add.at(x, np.minimum(((ev[m] - a) * tempo.BLOCK_HZ).astype(int), n - 1), w[m])
    return x


def fit_events(ev: np.ndarray, w: np.ndarray, a: float, b: float):
    """-> (bpm, t0) from stage 2 (tempo.estimate) on the events in [a, b), or None."""
    if ((ev >= a) & (ev < b)).sum() < 6:
        return None
    est = tempo.estimate(pulses(ev, w, a, b))
    return None if not np.isfinite(est.bpm) else (est.bpm, a + est.phase_ms / 1000.0)


def usable(est, lab) -> bool:
    if est is None:
        return False
    bpm, t0 = est
    if not tempo.tempo_ok(bpm, lab["bpm"])[1]:
        return False
    beat_ms = 60000.0 / lab["bpm"]
    d = ((t0 - lab["t0_s"]) * 1000.0) % beat_ms
    return min(d, beat_ms - d) <= gridmetrics.PHASE_OK_MS


def f_measure(ev, ref, spans):
    """Onset-style F against the grid; a detector that also marks offbeats loses precision."""
    ev = np.concatenate([ev[(ev >= a) & (ev < b)] for a, b in spans]) if spans else ev[:0]
    ref = np.concatenate([ref[(ref >= a) & (ref < b)] for a, b in spans]) if spans else ref[:0]
    if len(ev) == 0 or len(ref) == 0:
        return 0.0
    j = np.clip(np.searchsorted(ref, ev), 1, len(ref) - 1)
    d = np.minimum(np.abs(ref[j - 1] - ev), np.abs(ref[j] - ev))
    hit = np.sum(d <= F_TOL)
    p, r = hit / len(ev), min(hit, len(ref)) / len(ref)
    return float(2 * p * r / (p + r)) if p + r else 0.0


def score_song(ev, w, lab, dur):
    ref, ex = truth(lab, dur)
    edges = [0.0] + sorted({x for s in ex for x in s}) + [dur]
    spans = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a and included(ex, a, b)]
    out = {"F": f_measure(ev, ref, spans)}
    keep = np.zeros(len(ev), bool)
    for a, b in spans:
        keep |= (ev >= a) & (ev < b)
    if keep.sum() >= 8:
        # Period and phase from stage 2 over the whole song, excluded spans silent,
        # then one straight grid refined through the events.
        est = fit_events(ev[keep], w[keep], 0.0, dur)
        if est is not None:
            pk = np.stack([ev[keep], w[keep]], axis=1)
            f = fit_grid(pk, 60.0 / est[0], spans, est[1])
            if f is not None:
                bpm = 60.0 / f[1]
                r = bpm / lab["bpm"]
                out["global_exact"] = bool(abs(r - 1) < 0.001)
                out["global_octave"] = bool(any(abs(r / h - 1) < 0.001 for h in (1, 2, 0.5)))
                out["global_phase_ms"] = (abs(phase_diff_ms(bpm, f[0], lab["bpm"], lab["t0_s"], dur / 2))
                                          if out["global_octave"] else None)
    wins = []
    for a in np.arange(4.0, dur - WINDOW_S, WINDOW_S):
        if included(ex, a, a + WINDOW_S):
            wins.append(usable(fit_events(ev, w, a, a + WINDOW_S), lab))
    out["windows"] = wins
    return out


def events_ours(auto: dict):
    """Detections weighted by activation: the nearest thing to the activation stage 2 sees."""
    pk = np.asarray(auto["peaks"]).reshape(-1, 2)
    return pk[:, 0], pk[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", nargs="*", default=["songs/auto_cv_d0"])
    ap.add_argument("--conds", nargs="*", default=["clean", "device"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    labels = {p.stem: json.loads(p.read_text()) for p in (SONGS / "labels").glob("*.json")}
    labels = {k: v for k, v in labels.items() if v["status"] == "confirmed"}
    methods = {}
    for c in a.conds:
        d = SONGS / "beatnet" / c
        if d.exists():
            def bn(k, key, d=d):
                f = d / f"{k}.npz"
                if not f.exists() or key not in np.load(f).files:
                    return None
                t = np.load(f)[key][:, 0]
                return t, np.ones_like(t)
            methods[f"beatnet_offline_{c}"] = lambda k, bn=bn: bn(k, "offline")
            methods[f"beatnet_online_{c}"] = lambda k, bn=bn: bn(k, "online")
    for o in a.ours:
        methods[f"ours:{Path(o).name}"] = lambda k, o=o: (
            events_ours(json.loads((Path(o) / f"{k}.json").read_text()))
            if (Path(o) / f"{k}.json").exists() else None)
    # only songs every method covers, so the rows compare like with like
    common = [k for k in sorted(labels) if all(m(k) is not None for m in methods.values())]
    res = {"songs": len(common)}
    for name, m in methods.items():
        rows = [score_song(*m(k), labels[k], index[k]["duration_s"]) for k in common]
        w = np.concatenate([r["windows"] for r in rows]).astype(float)
        g = [r for r in rows if "global_exact" in r]
        ph = np.array([r["global_phase_ms"] for r in g if r.get("global_phase_ms") is not None])
        res[name] = {
            "F": float(np.mean([r["F"] for r in rows])),
            "window_usable": float(w.mean()), "windows": int(len(w)),
            "global_exact": float(np.mean([r["global_exact"] for r in g])) if g else None,
            "global_octave": float(np.mean([r["global_octave"] for r in g])) if g else None,
            "global_phase_median_ms": float(np.median(ph)) if len(ph) else None,
            "global_phase_within_10ms": float(np.mean(ph <= 10)) if len(ph) else None,
            "global_phase_within_25ms": float(np.mean(ph <= 25)) if len(ph) else None,
        }
        print(name, json.dumps(res[name]), flush=True)
    print(f"{len(common)} confirmed songs covered by every method")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
