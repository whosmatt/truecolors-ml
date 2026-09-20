"""End-to-end evaluation: BPM and phase, on real drum loops.

This is the actual deliverable — stage 1's job is only to feed it. The test set
is the 8,381 library drum loops with a duration-verified tempo, none of which
appear anywhere in training: real audio, known BPM, and because the loops are cut
sample-accurately to whole bars, a known beat grid as well.

Each loop is tiled from a random offset so the grid phase is not trivially zero,
then put through the same IR and coil-whine chain as the training corpus.
"""

import json
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

from dataset import augment, render
from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

BLOCK_MS = BLOCK_SAMPLES / SAMPLE_RATE * 1000.0
BLOCK_HZ = SAMPLE_RATE / BLOCK_SAMPLES
TAKE_S = 20.0
SETTLE_S = 4.0
BPM_RANGE = (55.0, 220.0)
# How much of the score a halved period must retain to be preferred. Loops repeat
# exactly, so the bar correlates as strongly as the beat; without this the
# estimator settles on whichever multiple happens to land on a whole block.
SUBHARMONIC_TOL = 0.6


def lag_range() -> tuple[int, int]:
    lo = int(np.floor(60.0 * BLOCK_HZ / BPM_RANGE[1]))
    hi = int(np.ceil(60.0 * BLOCK_HZ / BPM_RANGE[0]))
    return max(2, lo), hi


def bpm_of_lag(lag: float) -> float:
    return 60.0 * BLOCK_HZ / lag


@dataclass
class Estimate:
    bpm: float
    lag: float
    phase_ms: float
    strength: float


def autocorr(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation of a zero-mean signal, via FFT."""
    x = x - x.mean()
    n = 1 << (2 * len(x) - 1).bit_length()
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conj(f), n)[: len(x)]
    counts = np.arange(len(x), 0, -1)
    return ac / np.maximum(counts, 1)


def estimate(act: np.ndarray) -> Estimate:
    """Period and phase from an activation series.

    Period comes from the autocorrelation peak inside the tempo range. Phase then
    comes from correlating against a pulse train at that period, which is what
    turns a period into a usable grid.
    """
    lo, hi = lag_range()
    ac = autocorr(act)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return Estimate(float("nan"), float("nan"), float("nan"), 0.0)

    # Fractional lags, not integers. A period is rarely a whole number of blocks
    # (174 BPM is 32.33), and integer-lag autocorrelation smears for those while
    # an exact multiple further out does not — which makes the subharmonic score
    # higher than the true period.
    grid = np.arange(lo, hi + 1e-9, 0.05)
    interp = lambda L: np.interp(L, np.arange(len(ac)), ac)

    # Harmonic summation: a true period peaks at its multiples too.
    score = interp(grid).astype(np.float64)
    for k, w in ((2, 0.5), (3, 1 / 3)):
        score = score + w * np.where(grid * k < len(ac) - 1, interp(grid * k), 0.0)

    best = float(score.max())
    if best <= 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0.0)
    lag = float(grid[int(np.argmax(score))])

    # Prefer the fundamental over its subharmonics. On strictly periodic material
    # an exact multiple of the period can outscore the period itself: at 174 BPM
    # the beat is 32.33 blocks, so integer lags near it catch only two pairs in
    # three, while lag 97 (exactly three beats) catches every one. Accept a
    # halved or thirded period whenever it still explains most of the signal.
    score_at = lambda L: float(np.interp(L, grid, score)) if lo <= L <= hi else -np.inf
    for _ in range(3):
        for k in (2, 3):
            if score_at(lag / k) >= SUBHARMONIC_TOL * score_at(lag):
                lag = lag / k
                break
        else:
            break

    strength = float(interp(lag) / (np.abs(ac[0]) + 1e-12))

    # Joint refinement of period and phase by comb matching. Two reasons not to
    # fit a sinusoid: it finds the dominant Fourier component, which a
    # kick-on-1-3 / snare-on-2-4 pattern pulls away from the beat; and a period
    # that is off by even 0.1% drifts about a block across a 16 s window, so the
    # best single phase becomes a compromise. Searching both together removes it.
    x = act - act.mean()
    idx = np.arange(len(act))
    best_sum, lag, phase_blocks = -np.inf, lag, 0.0
    for cand in np.linspace(lag * 0.99, lag * 1.01, 41):
        phases = np.arange(0.0, cand, 0.1)
        beats = np.arange(int(len(act) / cand))
        pos = phases[:, None] + beats[None, :] * cand
        e = np.where(pos < len(act) - 1,
                     np.interp(np.clip(pos, 0, len(act) - 1), idx, x), 0.0).sum(axis=1)
        j = int(np.argmax(e))
        if e[j] > best_sum:
            best_sum, lag, phase_blocks = float(e[j]), float(cand), float(phases[j])

    return Estimate(bpm_of_lag(lag), float(lag), phase_blocks * BLOCK_MS, strength)


def grid_error_ms(est: Estimate, true_bpm: float, true_phase_ms: float) -> float:
    """Phase error against the true grid, folded into +/- half a beat.

    Measured at the estimated tempo: a grid that is right in period but wrong in
    phase is exactly as useless as one that is wrong in tempo.
    """
    beat_ms = 60000.0 / true_bpm
    d = (est.phase_ms - true_phase_ms) % beat_ms
    return float(min(d, beat_ms - d))


def tempo_ok(est_bpm: float, true_bpm: float, tol: float = 0.04) -> tuple[bool, bool]:
    """(exact within tol, correct up to an octave/triplet factor)."""
    if not np.isfinite(est_bpm) or est_bpm <= 0:
        return False, False
    exact = abs(est_bpm - true_bpm) / true_bpm <= tol
    octave = any(
        abs(est_bpm * r - true_bpm) / true_bpm <= tol
        for r in (1.0, 2.0, 0.5, 3.0, 1 / 3, 1.5, 2 / 3)
    )
    return exact, octave


def take_from_loop(row: dict, rng: np.random.Generator, notch_hz: int):
    """Tile a loop into a take, from a random offset, through IR and whine.

    -> (int16 pcm, true_phase_ms) where the phase is the first beat's position.
    """
    try:
        x, sr = sf.read(row["path"], dtype="float32", always_2d=True)
    except Exception:
        try:
            x, sr, _ = augment.able.read(row["path"])
        except Exception:
            return None, None
    if x.size == 0:
        return None, None
    m = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        m = soxr.resample(m, sr, SAMPLE_RATE, quality="VHQ").astype(np.float32)
    if m.size < SAMPLE_RATE // 2:
        return None, None

    off = int(rng.integers(m.size))
    need = int(TAKE_S * SAMPLE_RATE)
    tiled = np.resize(np.roll(m, -off), need)

    beat_samples = 60.0 / row["bpm"] * SAMPLE_RATE
    # The loop starts on a downbeat, so beats sit at multiples of beat_samples
    # from the loop start; rolling by `off` moves the grid by -off.
    phase_ms = ((-off) % beat_samples) / SAMPLE_RATE * 1000.0

    wet = augment.apply_ir(tiled)
    dbfs = float(rng.uniform(-30.0, -8.0))
    y = augment.finish(wet, notch_hz, rng, dbfs=dbfs)
    return (y * 32767.0).astype(np.int16), phase_ms


def test_loops(manifest: str, n: int, seed: int = 0) -> list[dict]:
    """Drum loops with a verified tempo, sampled deterministically."""
    rows = []
    with open(manifest) as fh:
        fh.readline()
        for line in fh:
            r = json.loads(line)
            if (
                r["kind"] == "loop"
                and r.get("bpm")
                and r.get("readable")
                and not r.get("discard")
                and "Drums" in set(r.get("families") or ())
            ):
                rows.append(r)
    rows.sort(key=lambda r: r["file_id"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]
