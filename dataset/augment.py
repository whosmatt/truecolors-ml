"""Make rendered audio resemble what the device actually hears.

The chain is physical, and the order matters:

    dry mix -> speaker/room/mic IR -> + coil whine -> ADC

The whine is added *after* the IR because the whine captures were recorded
through the same mic, so the mic's response is already in them.

Whine is injected at its measured absolute level, not relative to the music. The
coil is a fixed physical source: it does not get quieter when the music does,
which is exactly why it matters most at low SPL.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 48000
CAPTURES = Path("/mnt/c/Users/matt/truecolors/captures")
IR_PATH = Path("data/ir/mic_preliminary.wav")


@lru_cache(maxsize=4)
def load_ir(path: str = str(IR_PATH)) -> np.ndarray:
    """IR trimmed to start at the direct sound.

    The stored IR keeps 5 ms of pre-roll to hold deconvolution artefacts.
    Convolving with that pre-roll delays every onset by 4.27 ms (measured), so
    it is cut here: trimmed at the peak the shift is exactly zero.
    """
    ir, sr = sf.read(path, dtype="float64")
    if sr != SR:
        ir = soxr.resample(ir, sr, SR, quality="VHQ")
    ir = ir[int(np.argmax(np.abs(ir))) :]
    return (ir / np.abs(ir).max()).astype(np.float32)


@lru_cache(maxsize=8)
def load_whine(notch_hz: int, captures: str = str(CAPTURES)) -> np.ndarray:
    """A real 10 s capture of the device's own coil whine at this PWM setting.

    Real capture rather than synthesised comb lines: it carries the exact comb,
    the 5760 Hz mechanical resonance every setting excites, and the mic's own
    noise floor, with no synthesis error. Absolute scale is preserved.
    """
    p = Path(captures) / f"whine_{notch_hz}.wav"
    x, sr = sf.read(p, dtype="float64")
    x = x - x.mean()  # raw captures carry the mic's ~1057 LSB DC offset
    if abs(sr - SR) > 0.5:
        x = soxr.resample(x, sr, SR, quality="VHQ")
    return x.astype(np.float32)


@lru_cache(maxsize=2)
def load_floor(captures: str = str(CAPTURES)) -> np.ndarray:
    """Laser-off quiet capture: the mic's floor with no whine."""
    x, sr = sf.read(Path(captures) / "quiet_laser_off.wav", dtype="float64")
    x = x - x.mean()
    if abs(sr - SR) > 0.5:
        x = soxr.resample(x, sr, SR, quality="VHQ")
    return x.astype(np.float32)


@lru_cache(maxsize=4)
def _kernel_spectrum(key: tuple, nfft: int) -> np.ndarray:
    """rfft of the IR, cached: the same kernel is reused for every take."""
    return np.fft.rfft(_KERNELS[key], nfft)


_KERNELS: dict[tuple, np.ndarray] = {}


def apply_ir(x: np.ndarray, ir: np.ndarray | None = None) -> np.ndarray:
    """Convolve, preserving length and onset timing.

    Overlap-add in the frequency domain. Direct convolution of a 12 s take
    against a 24k-tap IR is 14 G MAC and dominated the whole build; this is the
    same arithmetic at O(n log m), with the kernel's spectrum computed once and
    reused across every take.
    """
    k = load_ir() if ir is None else ir
    n, m = len(x), len(k)
    if m == 0 or n == 0:
        return np.asarray(x, dtype=np.float32)
    if n < 4 * m:  # short takes: the transform costs more than it saves
        return np.convolve(x, k)[:n].astype(np.float32)

    nfft = 1 << max(11, (2 * m - 1).bit_length())
    hop = nfft - m + 1
    key = (k.shape[0], float(k[0]), float(k[-1]), float(k.sum()))
    _KERNELS.setdefault(key, k)
    K = _kernel_spectrum(key, nfft)

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

    `notch_hz` None means laser off, which still has the mic's floor.
    `gain` scales the whine only, for ablations; 1.0 is as measured.
    """
    bed = load_floor() if notch_hz is None else load_whine(notch_hz)
    if bed.size == 0:
        return x
    if len(x) > bed.size:
        bed = np.tile(bed, int(np.ceil(len(x) / bed.size)) + 1)
    start = int(rng.integers(bed.size - len(x))) if bed.size > len(x) else 0
    return (x + bed[start : start + len(x)] * gain).astype(np.float32)


def whine_lines(captures: str = str(CAPTURES)) -> dict[int, list[tuple[float, float]]]:
    """Measured comb lines, {pwm_hz: [(freq_hz, dB over quiet floor)]}."""
    raw = json.loads((Path(captures) / "whine_lines.json").read_text())
    return {int(k): [(float(f), float(d)) for f, d in v] for k, v in raw.items()}


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
    return finish(apply_ir(x), notch_hz, rng, dbfs=dbfs)
