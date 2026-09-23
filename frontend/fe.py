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


def _build(root: Path, defines: tuple[str, ...] = ()) -> Path:
    """Build one filter variant. Each gets its own .so, keyed by its defines."""
    inc = root / "components/audio/include"
    # The header counts: a bumped struct or constant with an untouched .c must
    # still rebuild.
    src = [root / "components/audio/frontend.c", _HERE / "shim.c", inc / "frontend.h"]
    tag = "".join("_" + d.lower().replace("fe_no_", "no") for d in sorted(defines))
    out = _HERE / "build" / f"libfe{tag}.so"
    out.parent.mkdir(exist_ok=True)
    if not out.exists() or out.stat().st_mtime < max(s.stat().st_mtime for s in src):
        subprocess.run(
            ["cc", "-O2", "-fPIC", "-shared", "-std=c99", f"-I{inc}",
             *[f"-D{d}" for d in defines],
             *[str(s) for s in src if s.suffix == ".c"], "-lm", "-o", str(out)],
            check=True,
        )
    return out


def _header_consts(root: Path) -> dict:
    """FE_* defines: plain integers and (1u << n) bit flags."""
    txt = (root / "components/audio/include/frontend.h").read_text()
    out = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"#define\s+(FE_\w+)\s+(\d+)\b", txt)
    }
    out.update(
        {
            m.group(1): 1 << int(m.group(2))
            for m in re.finditer(r"#define\s+(FE_\w+)\s+\(1u?\s*<<\s*(\d+)\)", txt)
        }
    )
    return out


def _load(defines: tuple[str, ...] = ()) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_build(_ROOT, defines)))
    for fn in ("fe_state_size", "fe_out_size"):
        getattr(lib, fn).restype = ctypes.c_size_t
    for fn in ("fe_spec_version", "fe_block_samples", "fe_sample_rate"):
        getattr(lib, fn).restype = ctypes.c_int
    lib.fe_variant.restype = ctypes.c_uint32
    lib.fe_init.argtypes = [ctypes.c_void_p]
    lib.fe_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return lib


_ROOT = truecolors_root()
_LIBS: dict[tuple[str, ...], ctypes.CDLL] = {}
_LIB = _load()
_LIBS[()] = _LIB
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

assert _LIB.fe_out_size() == DTYPE.itemsize, "fe_out_t layout changed, fix DTYPE"
assert _LIB.fe_spec_version() == SPEC_VERSION
assert _LIB.fe_block_samples() == BLOCK_SAMPLES
assert _LIB.fe_sample_rate() == SAMPLE_RATE

VARIANT_COMB = _C["FE_VARIANT_COMB"]
VARIANT_HICUT = _C["FE_VARIANT_HICUT"]


def variant_name(mask: int) -> str:
    on = [n for n, b in (("comb", VARIANT_COMB), ("hicut", VARIANT_HICUT)) if mask & b]
    return "+".join(on) if on else "none"


class Frontend:
    """One front-end instance. Stateful across run() calls.

    `hicut` selects a filter variant, compiled from the same firmware source with
    -DFE_NO_HICUT. The variant mask must be recorded alongside fe_spec_version: a
    model trained against one variant is not valid against another, and the
    mismatch is silent. v2 has no comb; v1 without it is bit-identical.
    """

    def __init__(self, comb: bool = False, hicut: bool = True):
        if comb:
            raise ValueError(f"FE_SPEC_VERSION {SPEC_VERSION} has no comb")
        defines = () if hicut else ("FE_NO_HICUT",)
        if defines not in _LIBS:
            _LIBS[defines] = _load(defines)
        self._lib = _LIBS[defines]
        self.variant = int(self._lib.fe_variant())
        self._st = ctypes.create_string_buffer(self._lib.fe_state_size())
        self._lib.fe_init(self._st)

    def run(self, samples: np.ndarray) -> np.ndarray:
        """int16 mono at SAMPLE_RATE -> (n_blocks,) structured array of features.

        A trailing partial block is dropped; feed whole takes, not slices, or
        the AGC state will not match what the device would have seen.
        """
        samples = np.ascontiguousarray(samples, dtype="<i2")
        n = samples.size // BLOCK_SAMPLES
        out = np.zeros(n, dtype=DTYPE)
        if n:
            self._lib.fe_run(self._st, samples.ctypes.data, n, out.ctypes.data)
        return out
