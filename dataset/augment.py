"""Make rendered audio resemble what the device actually hears.

The chain is physical, and the order matters:

    dry mix -> room IR -> device mic -> level -> + device self-noise -> ADC

Room and mic are one precomputed kernel per room channel (dataset/roomir.py),
drawn per take. The self-noise is added *after* the kernel because it was
recorded through the same mic, so the mic's response is already in it.

Self-noise is injected at its measured absolute level, not relative to the
music. Coil whine and the fan are fixed physical sources: they do not get
quieter when the music does, which is exactly why they matter most at low SPL.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 48000
KERNELS = Path("data/ir/kernels.npz")
# Controlled whine captures 2026-10-03 (after the hardware fix): slow RGB sweeps
# and the rainbow effect, laser on, fan running as it does in use.
SELF_NOISE = Path("data/whine/2026-10-03")
WHINE = {
    120: ("sweep120_a", "sweep120_b", "sweep120_c", "sweep120_d", "rainbow120"),
    240: ("sweep240_a", "sweep240_b"),
    480: ("sweep480_a", "sweep480_b"),
}
QUIET = ("quiet_pre", "quiet_post")  # laser off: mic floor and fan
XFADE = int(0.05 * SR)  # joins between capture pieces


@lru_cache(maxsize=1)
def _kernels() -> tuple[np.ndarray, ...]:
    z = np.load(KERNELS)
    d, o = z["data"], z["offsets"]
    return tuple(d[o[i] : o[i + 1]] for i in range(len(o) - 1))


def n_kernels() -> int:
    return len(_kernels())


@lru_cache(maxsize=None)
def _capture(name: str) -> np.ndarray:
    """A raw miccap capture, DC removed, absolute scale kept (1.0 = full scale)."""
    x = np.fromfile(SELF_NOISE / f"{name}.raw", dtype="<i2").astype(np.float64) / 32768.0
    return (x - x.mean()).astype(np.float32)


def bed(notch_hz: int | None, n: int, rng: np.random.Generator) -> np.ndarray:
    """n samples of real self-noise: random pieces of the captures for this PWM
    setting (None: laser off), joined with short crossfades."""
    names = QUIET if notch_hz is None else WHINE[notch_hz]
    out = np.zeros(n, np.float32)
    pos = 0
    while pos < n:
        c = _capture(names[int(rng.integers(len(names)))])
        L = min(len(c), n - pos + XFADE)
        s = int(rng.integers(len(c) - L + 1))
        seg = c[s : s + L].copy()
        if pos > 0:
            r = np.linspace(0.0, 1.0, min(XFADE, L), dtype=np.float32)
            seg[: len(r)] *= np.sqrt(r)
            out[pos - len(r) : pos] *= np.sqrt(r[::-1])
            start = pos - len(r)
        else:
            start = 0
        end = min(start + L, n)
        out[start:end] += seg[: end - start]
        pos = end
    return out


@lru_cache(maxsize=32)
def _kernel_spectrum(idx: int, nfft: int) -> np.ndarray:
    return np.fft.rfft(_kernels()[idx], nfft).astype(np.complex64)


def apply_ir(x: np.ndarray, rng: np.random.Generator, kernel: int | None = None) -> np.ndarray:
    """Convolve with one room+mic kernel, drawn from `rng` unless given;
    preserves length and onset timing (kernels start at the direct sound).

    Overlap-add in the frequency domain: direct convolution against a 1 s kernel
    would dominate the whole build.
    """
    idx = int(rng.integers(n_kernels())) if kernel is None else kernel
    k = _kernels()[idx]
    n, m = len(x), len(k)
    if n == 0:
        return np.asarray(x, dtype=np.float32)
    if n < 4 * m:  # short takes: the transform costs more than it saves
        return np.convolve(x, k)[:n].astype(np.float32)

    nfft = 1 << max(11, (2 * m - 1).bit_length())
    hop = nfft - m + 1
    K = _kernel_spectrum(idx, nfft)
    out = np.zeros(n + m - 1, dtype=np.float64)
    for i in range(0, n, hop):
        seg = x[i : i + hop]
        y = np.fft.irfft(np.fft.rfft(seg, nfft) * K, nfft)
        end = min(i + nfft, out.size)
        out[i:end] += y[: end - i]
    return out[:n].astype(np.float32)


def add_noise(
    x: np.ndarray, notch_hz: int | None, rng: np.random.Generator, gain: float = 1.0
) -> np.ndarray:
    """Add the device's own noise at its captured absolute level.

    `notch_hz` None means laser off, which still has the mic floor and the fan.
    `gain` scales the noise only, for ablations; 1.0 is as measured.
    """
    return (x + bed(notch_hz, len(x), rng) * gain).astype(np.float32)


def finish(
    wet: np.ndarray, notch_hz: int | None, rng: np.random.Generator, *, dbfs: float
) -> np.ndarray:
    """Level an already-convolved take and add the device's noise.

    Split from `apply_ir` because convolution is linear and its result does not
    depend on the notch setting or the level: one take can be convolved once and
    finished at several settings.

    Scaling happens before the noise is added, never after — scaling afterwards
    would drag the noise floor along with the music and undo the point of
    injecting it at an absolute level.
    """
    peak = float(np.abs(wet).max())
    if peak > 0:
        wet = wet / peak * 10 ** (dbfs / 20.0)
    return np.clip(add_noise(wet, notch_hz, rng), -1.0, 1.0)


def process(
    x: np.ndarray, notch_hz: int, rng: np.random.Generator, *, dbfs: float
) -> np.ndarray:
    """Full chain at a chosen playback level, for one-off use."""
    return finish(apply_ir(x, rng), notch_hz, rng, dbfs=dbfs)
