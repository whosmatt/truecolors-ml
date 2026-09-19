"""Render .alc drum clips with corpus one-shots, at exact note times.

This is the only training material with true onset ground truth: the note times
come from the MIDI, and each one-shot's onset is its post-trim start, which the
manifest measures at a median of 0.6 ms. Nothing here estimates an onset.

The front end's per-band AGC normalises against whatever is playing, so a bare
one-shot misleads it. Every render is therefore a full groove — percussion
included — repeated long enough for the AGC to settle, optionally over a
tempo-matched backing loop.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

from . import able
from .alc import Clip, Note

SR = 48000  # the front end's rate; see frontend.h FE_SAMPLE_RATE
RESAMPLE_Q = "VHQ"  # soxr one-shot compensates its own delay to <0.001 ms (2026-09-19)

# Rough kit balance before the AGC sees it. Absolute level barely matters, the
# relative one does.
CLASS_GAIN = {"kick": 1.0, "snare": 0.9, "hihat": 0.45, "perc": 0.6}
BACKING_GAIN = 0.35
AGC_SETTLE_S = 4.0  # measured: spl_db stabilises within ~3 s of a take (2026-09-19)
KEEP_S = 8.0        # usable audio kept after the settle window

# Only spl_db varies with absolute level: the AGC normalises everything else
# identically from -3 to -30 dBFS. Below about -45 dBFS the flux gate closes
# entirely, which is the front end's room-silence behaviour, so takes are drawn
# from a range that spans loud down to nearly-gated. (measured 2026-09-19)
LEVEL_DBFS = (-40.0, -6.0)


def _load(path: str) -> np.ndarray | None:
    """Mono, 48 kHz, leading silence trimmed, peak-normalised."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        try:
            x, sr, _ = able.read(path)
        except Exception:
            return None
    if x.size == 0:
        return None
    m = x.mean(axis=1)
    nz = np.flatnonzero(np.abs(x).max(axis=1) > 10 ** (-80 / 20.0))
    if nz.size == 0:
        return None
    m = m[int(nz[0]) :]
    if sr != SR:
        m = soxr.resample(m, sr, SR, quality=RESAMPLE_Q).astype(np.float32)
    peak = float(np.abs(m).max())
    return m / peak if peak > 0 else None


class Kit:
    """One drum kit: a stable sample per pad, drawn from the manifest.

    Keyed by pad rather than class, so a clip's two snare pads get two different
    samples — as they would in the rack the clip was written for.
    """

    def __init__(self, pools: dict[str, list[dict]], seed: str):
        self.pools = pools
        self.rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")
        )
        self._by_pad: dict[tuple[str, int], np.ndarray | None] = {}

    def voice(self, cls: str, midi: int) -> np.ndarray | None:
        key = (cls, midi)
        if key not in self._by_pad:
            pool = self.pools.get(cls) or []
            audio = None
            for _ in range(8):  # a few files decode to nothing; try again
                if not pool:
                    break
                row = pool[int(self.rng.integers(len(pool)))]
                audio = _load(row["path"])
                if audio is not None:
                    break
            self._by_pad[key] = audio
        return self._by_pad[key]


@dataclass
class Render:
    audio: np.ndarray        # float32 mono at SR
    onsets: np.ndarray       # sample index of each hit
    classes: np.ndarray      # matching class index into DETECTION order
    clip: str
    tempo: float
    backing: str | None

    def to_int16(self, dbfs: float = -1.0) -> np.ndarray:
        peak = float(np.abs(self.audio).max())
        x = self.audio / peak * 10 ** (dbfs / 20.0) if peak > 0 else self.audio
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def render(
    clip: Clip,
    kit: Kit,
    *,
    classes: tuple[str, ...],
    backing: tuple[np.ndarray, str] | None = None,
    min_seconds: float = AGC_SETTLE_S + KEEP_S,
) -> Render:
    loop_s = clip.duration_s
    repeats = max(1, int(np.ceil(min_seconds / loop_s)))
    total = int(np.ceil(loop_s * repeats * SR))
    tail = SR * 2
    buf = np.zeros(total + tail, dtype=np.float32)

    onsets, labels = [], []
    for rep in range(repeats):
        base = rep * loop_s
        for n in clip.notes:
            audio = kit.voice(n.cls, n.midi)
            if audio is None:
                continue
            at = int(round((base + clip.time_of(n)) * SR))
            if at >= total:
                continue
            gain = CLASS_GAIN.get(n.cls, 0.5) * (n.velocity / 127.0)
            end = min(at + audio.size, buf.size)
            buf[at:end] += audio[: end - at] * gain
            if n.cls in classes:
                onsets.append(at)
                labels.append(classes.index(n.cls))

    if backing is not None:
        bed, bed_name = backing
        tiled = np.resize(bed, buf.size) if bed.size else bed
        buf[: tiled.size] += tiled * BACKING_GAIN
    else:
        bed_name = None

    order = np.argsort(onsets) if onsets else np.array([], dtype=int)
    return Render(
        audio=buf[: total + SR // 2],
        onsets=np.asarray(onsets, dtype=np.int64)[order],
        classes=np.asarray(labels, dtype=np.int8)[order],
        clip=clip.name,
        tempo=clip.tempo,
        backing=bed_name,
    )


def pools_from_manifest(path: str) -> dict[str, list[dict]]:
    """Usable one-shots per class: readable, kept, and onset-trustworthy."""
    pools: dict[str, list[dict]] = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            r = json.loads(line)
            if r["kind"] != "one_shot" or r.get("discard") or not r.get("readable"):
                continue
            pools.setdefault(r["class"], []).append(r)
    return pools


def backing_pool(path: str) -> dict[int, list[dict]]:
    """Loops with a verified tempo, indexed by BPM, for tempo-matched beds."""
    out: dict[int, list[dict]] = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "loop" and r.get("bpm") and r.get("readable") and not r["class"]:
                out.setdefault(int(r["bpm"]), []).append(r)
    return out
