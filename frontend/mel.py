"""Device-identical mel flux: components/audio/melflux.c through ctypes.

The 16 mel-flux features of the two-stage model, as the firmware computes them.
Training corpora store 40 raw log-mel bands from the numpy prototype
(dataset.features.log_mel) so band layouts stay open; frontend.selftest checks
that the prototype's flux16 view matches this.
"""

import ctypes
import subprocess

import numpy as np

from .fe import _HERE, truecolors_root

BLOCK = 512


def _build(root):
    inc = root / "components/audio/include"
    src = [root / "components/audio/melflux.c", _HERE / "mel_shim.c",
           inc / "melflux.h", inc / "mel_filters.h"]
    out = _HERE / "build" / "libmelflux.so"
    out.parent.mkdir(exist_ok=True)
    if not out.exists() or out.stat().st_mtime < max(s.stat().st_mtime for s in src):
        subprocess.run(["cc", "-O2", "-fPIC", "-shared", "-std=c99", f"-I{inc}",
                        *[str(s) for s in src if s.suffix == ".c"], "-lm", "-o", str(out)],
                       check=True)
    return out


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(_build(truecolors_root())))
        _LIB.melflux_state_size.restype = ctypes.c_size_t
        _LIB.melflux_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return _LIB


class MelFlux:
    """One stream from boot; run() continues it."""

    def __init__(self):
        lib = _lib()
        self._st = ctypes.create_string_buffer(lib.melflux_state_size())
        lib.melflux_init(self._st)
        self.bands = lib.melflux_bands()

    def run(self, samples: np.ndarray) -> np.ndarray:
        """int16 mono at 48 kHz -> (n_blocks, 16) float32; a trailing partial block is dropped."""
        s = np.ascontiguousarray(samples, dtype="<i2")
        n = s.size // BLOCK
        out = np.zeros((n, self.bands), np.float32)
        if n:
            _lib().melflux_run(self._st, s.ctypes.data, n, out.ctypes.data)
        return out
