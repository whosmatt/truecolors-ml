"""Loop tempo from the filename, verified against duration.

Live's index stores no tempo, and .asd sidecars exist for only 5.6% of drum
loops. Filenames declare it often enough, but a bare number is as likely to be a
pack ID as a tempo, so duration does the disambiguating: at the true tempo a
loop's length lands on a whole number of bars.
"""

import re
from dataclasses import dataclass

BPM_MIN, BPM_MAX = 50, 220
TOL = 0.005  # loops that declare bpm verify at p75 bar-count error 3e-4 (2026-09-19)
BAR_COUNTS = (1, 2, 4, 8, 16)  # power-of-two only: false accepts 0.2% vs 6.0% for any 4-beat multiple

_EXPLICIT = (
    re.compile(r"(?<!\d)(\d{2,3})\s*[-_ ]?\s*bpm", re.I),
    re.compile(r"bpm\s*[-_ ]?\s*(\d{2,3})(?!\d)", re.I),
)
_BARE = re.compile(r"(?<!\d)(\d{2,3})(?!\d)")


@dataclass
class Tempo:
    bpm: int
    bars: int
    beats: int
    source: str  # explicit | filename | folder


def _beats_fit(duration_s: float, bpm: int) -> int | None:
    """Whole-bar beat count this tempo implies, or None."""
    b = duration_s * bpm / 60.0
    n = round(b)
    if n < 4 or n % 4 or abs(b - n) / n > TOL:
        return None
    return n if n // 4 in BAR_COUNTS else None


def _candidates(text: str) -> list[int]:
    return sorted({int(m) for m in _BARE.findall(text) if BPM_MIN <= int(m) <= BPM_MAX})


def _explicit(text: str) -> int | None:
    for rx in _EXPLICIT:
        m = rx.search(text)
        if m and BPM_MIN <= int(m.group(1)) <= BPM_MAX:
            return int(m.group(1))
    return None


def tempo_of(name: str, folders: str, duration_s: float) -> Tempo | None:
    """Filename first, folder names as fallback. Ambiguity is rejected, not guessed."""
    declared = _explicit(name) or _explicit(folders)
    if declared is not None:
        n = _beats_fit(duration_s, declared)
        return Tempo(declared, n // 4, n, "explicit") if n else None

    for text, source in ((name, "filename"), (folders, "folder")):
        fits = [(c, _beats_fit(duration_s, c)) for c in _candidates(text)]
        fits = [(c, n) for c, n in fits if n]
        if len(fits) == 1:
            bpm, n = fits[0]
            return Tempo(bpm, n // 4, n, source)
        if len(fits) > 1:
            return None
    return None
