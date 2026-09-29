"""Self-supervised beat grids for whole songs.

The model's lock is intermittent across a song but precise where it holds. On
constant-tempo music, locks far apart pin one grid across the whole track: the
fitted period's error shrinks with the span, so the grid can be better than any
single local estimate, and it labels the sections the model missed as well.

Pipeline per song: decode, the device chain (IR, whine, level) and the firmware
front end, the model's beat activation, then fixed segments that each get a
local grid. Segments with a tight, well-covered local fit are locks; the global
grid grows from the longest run of locks outwards, one agreeing lock at a time.

    python -m dataset.songgrid                 # every downloaded song
    python -m dataset.songgrid --ids 4V4xt...  # just these

Writes songs/auto/<id>.json. The audio timeline is the downloaded file's, and
labels are beat times on it, so a new FE or IR only needs a rebuild, not a relabel.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE, SPEC_VERSION

SONGS = Path("songs")
MODEL = Path("runs/m_cand_s0")
BLOCK_S = BLOCK_SAMPLES / SAMPLE_RATE

SEG_S = 12.0
PEAK_MIN = 0.3        # beat activation a peak must reach to count as a detection
MATCH_BEATS = 0.25    # a peak within this fraction of a beat of a grid beat matches it
MATCH_MAX_MS = 60.0
SOLID_COVERAGE = 0.75  # locks: most grid beats have a detection ...
SOLID_ERR_MS = 15.0    # ... and they sit tightly on the grid
AGREE_MS = 20.0        # a lock agrees with the global grid when its median offset is this close
DRIFT_MAX = 0.002      # period change start to end beyond which tempo is not constant
UNCERTAIN_MAX_MS = 2.0
# No snapping to whole tempos: hand labels confirmed fits like 128.026 by ear;
# the chain onto YouTube can stretch a song slightly (2026-09-27).


def decode(path: Path) -> np.ndarray:
    """Any audio file -> mono float32 at the front end's rate."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def device_features(m: np.ndarray, seed: int) -> np.ndarray:
    from . import augment, features, render

    rng = np.random.default_rng(seed)
    wet = augment.apply_ir(m)
    y = augment.finish(wet, 240, rng, dbfs=float(np.mean(render.LEVEL_DBFS)))
    pcm = (y * 32767.0).astype(np.int16)
    X = features.featurise(pcm, hicut=True)
    if not features.MEL:  # mel models need the columns; others ignore them
        X = np.concatenate([X, features.log_mel(pcm)], axis=1)
    return X


class Model:
    def __init__(self, run: Path = MODEL):
        from train import gpu  # noqa: F401  must precede keras
        import keras

        from train import scoreboard

        self.run = run
        self.m = keras.models.load_model(run / "model.keras", compile=False)
        nz = np.load(run / "norm.npz")
        self.mean, self.scale = nz["mean"], nz["scale"]
        self.spec = scoreboard.run_spec(run)
        info = json.loads((run / "result.json").read_text())
        self.mel = info.get("mel")
        assert not info.get("cascade"), "cascade models need model A run alongside"

    def __call__(self, X: np.ndarray):
        """-> per block (beat, beat_offset, music, hit (n, 4)); zero before the lookback fills."""
        from train import data, scoreboard

        n = len(X)
        z = np.zeros((n, 1), dtype=np.float32)
        s = data.view(data.Split(X, z, z, np.zeros(n, dtype=np.int32), np.arange(n)), self.mel)
        a, out, c = scoreboard.activation(self.m, s, self.mean, self.scale, self.spec)
        off, mus = np.zeros(n, np.float32), np.zeros(n, np.float32)
        hit = np.zeros((n, 4), np.float32)
        off[c] = out["beat_offset"][:, 0]
        hit[c] = out["hit"]
        if "music" in out:
            mus[c] = out["music"][:, 0]
        return a, off, mus, hit


def peaks(act: np.ndarray, off: np.ndarray) -> np.ndarray:
    """-> (n, 2) [time s, activation] of local maxima above PEAK_MIN."""
    a = act
    i = np.flatnonzero((a[1:-1] >= PEAK_MIN) & (a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:])) + 1
    return np.stack([(i + off[i]) * BLOCK_S, a[i]], axis=1)


HIT_CLASSES = ("kick", "snare")  # hit head columns 0, 1; 2 is hihat, 3 none


def on_off(hit: np.ndarray, bpm: float, t0: float, spans) -> dict:
    """Mean kick and snare activation at grid beats and at the half-beat points
    between them, peak within +-1 block. Kicks and backbeat snares sit on beats;
    offbeat hats, which the beat head can latch onto, sit on the half beats."""
    T = 60.0 / bpm
    out = {}
    for c, name in enumerate(HIT_CLASSES):
        for key, shift in (("on", 0.0), ("off", 0.5)):
            v = []
            for s, e in spans:
                k = np.arange(np.ceil((s - t0) / T - shift), np.ceil((e - t0) / T - shift))
                b = ((t0 + (k + shift) * T) / BLOCK_S).astype(int)
                b = b[(b >= 1) & (b < len(hit) - 1)]
                v.append(np.max(np.stack([hit[b - 1, c], hit[b, c], hit[b + 1, c]]), axis=0))
            v = np.concatenate(v) if v else np.zeros(0)
            out[f"{name}_{key}"] = float(v.mean()) if len(v) else 0.0
    return out


def match(pk: np.ndarray, bpm: float, t0: float, start: float, end: float):
    """Grid beats in [start, end) against their nearest detection.

    -> (grid times, signed residual s or nan when unmatched, beat index)."""
    T = 60.0 / bpm
    k = np.arange(np.ceil((start - t0) / T), np.ceil((end - t0) / T))
    g = t0 + k * T
    if len(g) == 0 or len(pk) == 0:
        return g, np.full(len(g), np.nan), k
    t = pk[:, 0]
    j = np.clip(np.searchsorted(t, g), 1, len(t) - 1)
    near = np.where(np.abs(t[j - 1] - g) <= np.abs(t[j] - g), t[j - 1], t[j])
    d = near - g
    tol = min(MATCH_BEATS * T, MATCH_MAX_MS / 1000.0)
    return g, np.where(np.abs(d) <= tol, d, np.nan), k


def grid_error(pk: np.ndarray, bpm: float, t0: float, start: float, end: float) -> dict:
    """Fit of one grid over one span: RMS and median residual of matched beats, in
    ms, and the fraction of grid beats matched. A positive offset means the
    detections come after the grid: nudge the grid later by that much."""
    g, d, _ = match(pk, bpm, t0, start, end)
    ok = np.isfinite(d)
    if not ok.any():
        return {"err_ms": None, "offset_ms": None, "coverage": 0.0, "beats": int(len(g))}
    return {"err_ms": float(np.sqrt(np.mean(d[ok] ** 2)) * 1000),
            "offset_ms": float(np.median(d[ok]) * 1000),
            "coverage": float(ok.mean()), "beats": int(len(g))}


def evaluate(auto: dict, bpm: float, t0: float) -> dict:
    """Any grid (the model's or a hand-corrected one) against the song's detections."""
    pk = np.asarray(auto["peaks"], dtype=np.float64).reshape(-1, 2)
    secs = [grid_error(pk, bpm, t0, s["start_s"], s["end_s"]) for s in auto["sections"]]
    return {"total": grid_error(pk, bpm, t0, 0.0, auto["duration_s"]), "sections": secs}


def _lsq(t: np.ndarray, k: np.ndarray):
    A = np.stack([np.ones_like(k), k], axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    r = t - A @ coef
    var = float(r @ r) / max(len(t) - 2, 1)
    return coef, var * np.linalg.inv(A.T @ A), r


def fit_grid(pk: np.ndarray, T: float, spans: list[tuple[float, float]], t_ref: float):
    """One straight grid through the detections in `spans`, beats numbered from t_ref.

    -> (t0, T, covariance, residuals s, beat indices)."""
    t0 = t_ref
    for _ in range(4):
        tt, kk = [], []
        for s, e in spans:
            g, d, k = match(pk, 60.0 / T, t0, s, e)
            ok = np.isfinite(d)
            tt.append(g[ok] + d[ok])
            kk.append(k[ok])
        t, k = np.concatenate(tt), np.concatenate(kk).astype(np.float64)
        if len(t) < 8:
            return None
        (t0, T), cov, r = _lsq(t, k)
    return float(t0), float(T), cov, r, k


def segments(duration: float) -> list[tuple[float, float]]:
    n = max(1, int(round(duration / SEG_S)))
    edges = np.linspace(0.0, duration, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def comb_phase(a: np.ndarray, lag: float) -> tuple[float, float]:
    """Best (lag, phase) in blocks for a pulse train within 1% of `lag`, as in
    tempo.estimate's refinement but at a period chosen from outside."""
    x = a - a.mean()
    idx = np.arange(len(a))
    best, out = -np.inf, (lag, 0.0)
    for cand in np.linspace(lag * 0.99, lag * 1.01, 41):
        ph = np.arange(0.0, cand, 0.1)
        pos = ph[:, None] + np.arange(int(len(a) / cand))[None, :] * cand
        e = np.where(pos < len(a) - 1,
                     np.interp(np.clip(pos, 0, len(a) - 1), idx, x), 0.0).sum(axis=1)
        j = int(np.argmax(e))
        if e[j] > best:
            best, out = float(e[j]), (float(cand), float(ph[j]))
    return out


def is_solid(ge: dict) -> bool:
    return bool(ge["coverage"] >= SOLID_COVERAGE and ge["err_ms"] is not None
                and ge["err_ms"] <= SOLID_ERR_MS)


HARMONIC = (1.0, 2.0, 0.5, 4 / 3, 3 / 4, 3 / 2, 2 / 3, 3.0, 1 / 3)


def analyse(act, off, mus, hit, duration: float, spotify_bpm: float | None = None) -> dict:
    from train import tempo

    pk = peaks(act, off)
    secs = []
    # Pass 1: each segment's own tempo. 12 s is short enough that the estimator
    # sometimes lands on 4/3 or 3/2 of the beat, so this only sets the consensus.
    for s, e in segments(duration):
        b0, b1 = int(s / BLOCK_S), int(e / BLOCK_S)
        row = {"start_s": float(s), "end_s": float(e),
               "music": float(np.median(mus[b0:b1])) if b1 > b0 else 0.0,
               "local_bpm": None, "local_err_ms": None, "local_coverage": 0.0,
               "solid": False, "agrees": None, "off_tempo": False}
        a = act[b0:b1]
        if len(a) > 64 and a.max() >= PEAK_MIN:
            est = tempo.estimate(a)
            if np.isfinite(est.bpm):
                row["free_bpm"] = est.bpm
                row["free_solid"] = is_solid(grid_error(pk, est.bpm, s + est.phase_ms / 1000.0, s, e))
        secs.append(row)

    out = {"duration_s": duration, "peaks": np.round(pk, 5).tolist(), "sections": secs,
           "fit": None, "hits": {}}
    for c, name in enumerate(HIT_CLASSES):
        h = hit[:, c]
        i = np.flatnonzero((h[1:-1] >= PEAK_MIN) & (h[1:-1] >= h[:-2]) & (h[1:-1] > h[2:])) + 1
        out["hits"][name] = np.round(np.stack([i * BLOCK_S, h[i]], axis=1), 4).tolist()
    free = [60.0 / r["free_bpm"] for r in secs if r.get("free_solid")] or \
           [60.0 / r["free_bpm"] for r in secs if r.get("free_bpm")]
    T = None
    if free:
        lags = np.array(free)
        ref = float(np.median(lags))
        T = float(np.median(lags * 2.0 ** np.round(np.log2(ref / lags))))

    # Pass 2: every segment at the consensus period, phase and +-1% period free.
    if T is not None:
        for row in secs:
            b0, b1 = int(row["start_s"] / BLOCK_S), int(row["end_s"] / BLOCK_S)
            a = act[b0:b1]
            if len(a) <= 64 or a.max() < PEAK_MIN:
                continue
            lag, ph = comb_phase(a, T / BLOCK_S)
            bpm, t0 = 60.0 / (lag * BLOCK_S), row["start_s"] + ph * BLOCK_S
            ge = grid_error(pk, bpm, t0, row["start_s"], row["end_s"])
            row.update(local_bpm=bpm, local_t0_s=t0, local_err_ms=ge["err_ms"],
                       local_coverage=ge["coverage"], solid=is_solid(ge))
            # Locked at a tempo no simple ratio away from the consensus: a tempo change.
            if row.get("free_solid") and not row["solid"]:
                ratio = row["free_bpm"] * T / 60.0
                row["off_tempo"] = not any(abs(ratio / h - 1) < 0.02 for h in HARMONIC)

    solid = [i for i, r in enumerate(secs) if r["solid"]]
    reasons = []
    if len(solid) >= 2:
        # Seed: the longest run of consecutive locks.
        runs, cur = [], [solid[0]]
        for i in solid[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        seed = max(runs, key=len)
        inc = list(seed)
        f = fit_grid(pk, T, [(secs[i]["start_s"], secs[i]["end_s"]) for i in inc],
                     secs[seed[0]]["local_t0_s"])
        rest = [i for i in solid if i not in inc]
        while f is not None and rest:
            t0, T = f[0], f[1]
            lo, hi = min(inc), max(inc)
            i = min(rest, key=lambda j: min(abs(j - lo), abs(j - hi)))
            rest.remove(i)
            ge = grid_error(pk, 60.0 / T, t0, secs[i]["start_s"], secs[i]["end_s"])
            agrees = (ge["coverage"] >= 0.6 and ge["offset_ms"] is not None
                      and abs(ge["offset_ms"]) <= AGREE_MS)
            secs[i]["agrees"] = bool(agrees)
            if agrees:
                trial = fit_grid(pk, T, [(secs[j]["start_s"], secs[j]["end_s"])
                                         for j in sorted(inc + [i])], t0)
                if trial is not None:
                    inc.append(i)
                    f = trial
        for i in seed:
            secs[i]["agrees"] = True

        if f is not None:
            t0, T, cov, r, k = f
            spans = [(secs[i]["start_s"], secs[i]["end_s"]) for i in sorted(inc)]
            kk = np.arange(np.ceil(-t0 / T), np.ceil((duration - t0) / T))
            sig = np.sqrt(np.maximum(cov[0, 0] + kk**2 * cov[1, 1] + 2 * kk * cov[0, 1], 0))
            q = np.polyfit(k, t0 + k * T + r, 2)
            drift = float(2 * q[0] * (k.max() - k.min()) / T)
            span = spans[-1][1] - spans[0][0]
            fit = {"bpm_free": 60.0 / T, "err_ms_free": float(np.sqrt(np.mean(r**2)) * 1000),
                   "uncertainty_ms": float(sig.max() * 1000), "drift": drift,
                   "span_s": float(span), "locks_used": len(inc)}
            t_first = t0 + np.ceil(-t0 / T) * T  # first beat at or after the file start
            fit.update(bpm=60.0 / T, t0_s=float(t_first), err_ms=float(np.sqrt(np.mean(r**2)) * 1000))
            # Hints only: tested on the first hand labels, neither the hit head
            # nor band flux told beats from offbeats reliably (2026-09-27).
            fit.update(on_off(hit, fit["bpm"], t_first, [(0.0, duration)]))
            fit["spotify_ratio"], fit["tempo_hint"] = None, None
            if spotify_bpm:
                ratio = fit["bpm"] / spotify_bpm
                fit["spotify_ratio"] = float(ratio)
                octave = 2.0 ** np.round(np.log2(ratio))
                if abs(ratio / octave - 1) > 0.02:
                    for num, den in ((3, 2), (2, 3), (4, 3), (3, 4)):
                        o = ratio / (num / den)
                        if abs(o / 2.0 ** np.round(np.log2(o)) - 1) < 0.02:
                            b = fit["bpm"] * den / num * 2.0 ** -np.round(np.log2(o))
                            fit["tempo_hint"] = f"{b:.2f} (fit is {num}/{den} of Spotify)"
                            fit["tempo_hint_bpm"] = float(b)
                            break
            out["fit"] = fit
            for row in secs:
                ge = grid_error(pk, fit["bpm"], t_first, row["start_s"], row["end_s"])
                row.update(err_ms=ge["err_ms"], offset_ms=ge["offset_ms"], coverage=ge["coverage"])

    # Filters.
    n_solid = len(solid)
    fit = out["fit"]
    n_dis = sum(1 for r in secs if r["agrees"] is False or r["off_tempo"])
    if n_solid < 2 or n_solid < 0.15 * len(secs) or fit is None:
        auto = "non_rhythmic"
        reasons.append(f"{n_solid} of {len(secs)} segments locked")
    elif n_dis >= max(2, 0.25 * n_solid) or abs(fit["drift"]) > DRIFT_MAX:
        auto = "variable_tempo"
        reasons.append(f"{n_dis} segments disagree or off tempo ({n_solid} locks), drift {fit['drift']*100:+.2f}%")
    elif (fit["span_s"] < 0.5 * duration or fit["uncertainty_ms"] > UNCERTAIN_MAX_MS
          or fit["err_ms"] > SOLID_ERR_MS):
        auto = "insufficient_lock"
        reasons.append(f"locks span {fit['span_s']:.0f} of {duration:.0f} s, "
                       f"err {fit['err_ms']:.1f} ms, uncertainty {fit['uncertainty_ms']:.1f} ms")
    elif fit["tempo_hint"]:
        auto = "check_tempo"
        reasons.append(f"tempo hint {fit['tempo_hint']}")
    else:
        auto = "accept"
        reasons.append(f"{fit['locks_used']} locks over {fit['span_s']:.0f} s, "
                       f"err {fit['err_ms']:.1f} ms, uncertainty {fit['uncertainty_ms']:.2f} ms")
    if fit and fit["tempo_hint"] and auto != "check_tempo":
        reasons.append(f"tempo hint {fit['tempo_hint']}")
    out["verdict"] = {"auto": auto, "reasons": reasons, "needs_review": auto != "accept"}
    out["verdict"]["needs_review"] = True  # first pass: a human checks every filter decision
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run", type=Path, default=MODEL)
    ap.add_argument("--out-dir", type=Path, default=SONGS / "auto",
                    help="the labeller reads songs/auto; write other models elsewhere")
    a = ap.parse_args()
    index = [json.loads(l) for l in (SONGS / "index.jsonl").read_text().splitlines() if l.strip()]
    todo = [r for r in index if r.get("status") == "ok" and (a.ids is None or r["id"] in a.ids)]
    model = Model(a.run)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for n, row in enumerate(todo):
        dst = a.out_dir / f"{row['id']}.json"
        if dst.exists() and not a.force:
            continue
        t = time.time()
        m = decode(SONGS / row["audio"])
        X = device_features(m, seed=int.from_bytes(row["id"].encode()[:4], "big"))
        act, off, mus, hit = model(X)
        res = analyse(act, off, mus, hit, len(m) / SAMPLE_RATE, row.get("spotify_tempo"))
        res.update(id=row["id"], analysed=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   model=str(a.run), fe_spec_version=SPEC_VERSION)
        tmp = dst.with_suffix(".tmp")  # the labeller may be reading it
        tmp.write_text(json.dumps(res))
        tmp.replace(dst)
        fit = res["fit"] or {}
        print(f"{n+1}/{len(todo)} {row['artist'][:20]:20} {row['title'][:28]:28} "
              f"{res['verdict']['auto']:17} bpm {fit.get('bpm', 0):6.2f} "
              f"(spotify {row.get('spotify_tempo', 0):6.1f})  {time.time()-t:.1f}s", flush=True)
    print(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
