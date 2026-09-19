"""Per-file audio measurements: geometry, content hash, onset offset.

The onset rule comes from the corpus: ~30% of one-shots carry leading digital
silence whose p90 is exactly 50.00 ms (pack export convention), which is five
front-end blocks of labelling error if sample 0 is taken as the onset. Trim it
first; the residual attack offset then has a median of 0.5-1.25 ms.
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf

from . import able

SILENCE_DBFS = -80.0   # pack pads are exact zeros, so the threshold is not critical
ATTACK_REL_DB = -20.0  # attack = first crossing of this, relative to post-trim peak
RMS_WIN_S = 0.005
ATTACK_DISCARD_MS = 5.0  # corpus p90 = 4.0-6.1 ms; discards 8.8-13.5% by class (2026-09-19)
HASH_HEAD_BYTES = 256 * 1024


@dataclass
class Probe:
    samplerate: int
    channels: int
    frames: int
    duration_s: float
    subtype: str
    encrypted: bool
    file_size: int
    head_sha256: str
    lead_silence_ms: float | None = None
    attack_offset_ms: float | None = None
    discard: bool = False
    discard_reason: str | None = None

    def asdict(self) -> dict:
        return asdict(self)


def _head_hash(path: str, size: int) -> str:
    """sha256 of file size + first 256 KB. Not a full-content hash: loops run to
    megabytes and the corpus is tens of GB. Enough to spot duplicates alongside
    size and the Live embedding."""
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(HASH_HEAD_BYTES))
    return h.hexdigest()


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    w = max(1, int(sr * RMS_WIN_S))
    return np.sqrt(np.convolve(x * x, np.ones(w) / w, mode="full")[: len(x)])


def probe(path: str, *, onset: bool) -> Probe | None:
    """Header geometry always; decode and measure the onset only when asked.

    Returns None if the file cannot be read or holds no signal.
    """
    encrypted = False
    try:
        info = sf.info(path)
        rate, frames, channels = info.samplerate, info.frames, info.channels
        subtype = info.subtype or ""
    except Exception:
        # Ableton's DRM-protected AIFC; libsndfile rejects the COMM tag.
        try:
            frames, channels, rate, bits = able.info(path)
        except Exception:
            return None
        subtype, encrypted = f"PCM_{bits}", True
    if not rate or not frames:
        return None

    import os

    size = os.path.getsize(path)
    p = Probe(
        samplerate=rate,
        channels=channels,
        frames=frames,
        duration_s=frames / rate,
        subtype=subtype,
        encrypted=encrypted,
        file_size=size,
        head_sha256=_head_hash(path, size),
    )
    if not onset:
        return p

    try:
        x, sr = (able.read(path)[:2] if encrypted
                 else sf.read(path, dtype="float32", always_2d=True))
    except Exception:
        p.discard, p.discard_reason = True, "decode_failed"
        return p
    if x.shape[0] < 64:
        p.discard, p.discard_reason = True, "too_short"
        return p

    # Trim on the loudest channel, so an out-of-phase stereo pair cannot read as
    # silence; measure the attack on the mono mix.
    lead = np.flatnonzero(np.abs(x).max(axis=1) > 10 ** (SILENCE_DBFS / 20.0))
    if lead.size == 0:
        p.discard, p.discard_reason = True, "silent"
        return p
    start = int(lead[0])
    p.lead_silence_ms = start / sr * 1000.0

    env = _envelope(x[start:].mean(axis=1), sr)
    peak = float(env.max())
    if peak <= 0.0:
        p.discard, p.discard_reason = True, "silent"
        return p
    cross = int(np.argmax(env >= peak * 10 ** (ATTACK_REL_DB / 20.0)))
    p.attack_offset_ms = cross / sr * 1000.0
    if p.attack_offset_ms > ATTACK_DISCARD_MS:
        p.discard, p.discard_reason = True, "slow_attack"
    return p
