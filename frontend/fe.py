"""Device-identical audio front end.

Compiles components/audio/frontend.c from the truecolors checkout and uses it
through ctypes. Use this for training features, don't reimplement in numpy :)
"""

import ctypes
import os
import re
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def truecolors_root() -> Path:
    """Locate the firmware checkout: $TRUECOLORS_ROOT, the submodule, or a sibling."""
    cands = []
    if os.environ.get("TRUECOLORS_ROOT"):
        cands.append(Path(os.environ["TRUECOLORS_ROOT"]))
    cands += [_HERE.parent / "truecolors", _HERE.parent.parent]
    for c in cands:
        if (c / "components/audio/frontend.c").is_file():
            return c.resolve()
    raise FileNotFoundError(
        "no truecolors checkout found (set TRUECOLORS_ROOT); tried: "
        + ", ".join(str(c) for c in cands)
    )


def _build(root: Path) -> Path:
    src = [root / "components/audio/frontend.c", _HERE / "shim.c"]
    inc = root / "components/audio/include"
    out = _HERE / "build" / "libfe.so"
    out.parent.mkdir(exist_ok=True)
    if not out.exists() or out.stat().st_mtime < max(s.stat().st_mtime for s in src):
        subprocess.run(
            ["cc", "-O2", "-fPIC", "-shared", "-std=c99", f"-I{inc}",
             *[str(s) for s in src], "-lm", "-o", str(out)],
            check=True,
        )
    return out


def _header_consts(root: Path) -> dict:
    txt = (root / "components/audio/include/frontend.h").read_text()
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"#define\s+(FE_\w+)\s+(\d+)\b", txt)
    }


_ROOT = truecolors_root()
_LIB = ctypes.CDLL(str(_build(_ROOT)))
_C = _header_consts(_ROOT)

SPEC_VERSION = _C["FE_SPEC_VERSION"]
SAMPLE_RATE = _C["FE_SAMPLE_RATE"]
BLOCK_SAMPLES = _C["FE_BLOCK_SAMPLES"]
BLOCK_HZ = SAMPLE_RATE / BLOCK_SAMPLES

# Mirrors fe_out_t
DTYPE = np.dtype([
    ("level", "<f4"),
    ("bands", "<f4", 3),
    ("rms", "<f4"),
    ("flux", "<f4", 3),
    ("fund_rms", "<f4"),
    ("mid_flux", "<f4"),
    ("treble_flux", "<f4"),
    ("spl_db", "<f4"),
])

for _fn in ("fe_state_size", "fe_out_size"):
    getattr(_LIB, _fn).restype = ctypes.c_size_t
for _fn in ("fe_spec_version", "fe_block_samples", "fe_sample_rate"):
    getattr(_LIB, _fn).restype = ctypes.c_int

assert _LIB.fe_out_size() == DTYPE.itemsize, "fe_out_t layout changed, fix DTYPE"
assert _LIB.fe_spec_version() == SPEC_VERSION
assert _LIB.fe_block_samples() == BLOCK_SAMPLES
assert _LIB.fe_sample_rate() == SAMPLE_RATE

_LIB.fe_init.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_LIB.fe_set_notch_hz.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_LIB.fe_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


class Frontend:
    """One front-end instance. Stateful across run() calls"""

    def __init__(self, notch_hz: int = 480):
        self._st = ctypes.create_string_buffer(_LIB.fe_state_size())
        _LIB.fe_init(self._st, notch_hz)

    def set_notch_hz(self, hz: int):
        _LIB.fe_set_notch_hz(self._st, hz)

    def run(self, samples: np.ndarray) -> np.ndarray:
        """int16 mono at SAMPLE_RATE -> (n_blocks,) structured array of features.

        A trailing partial block is dropped; feed whole takes, not slices, or
        the AGC state will not match what the device would have seen.
        """
        samples = np.ascontiguousarray(samples, dtype="<i2")
        n = samples.size // BLOCK_SAMPLES
        out = np.zeros(n, dtype=DTYPE)
        if n:
            _LIB.fe_run(self._st, samples.ctypes.data, n, out.ctypes.data)
        return out
