"""Make CUDA visible to TensorFlow before it is imported.

`tensorflow[and-cuda]` installs its CUDA libraries under the venv's `nvidia/`
tree without putting them on the loader path, and WSL keeps the driver's
`libcuda.so` in /usr/lib/wsl/lib. Missing either, TF reports "Cannot dlopen some
GPU libraries" and falls back to CPU silently, which here is a 22x slowdown that
looks like nothing is wrong.

LD_LIBRARY_PATH is read at process start, so setting it from Python is too late.
Instead each library is loaded with RTLD_GLOBAL up front: a soname already in the
global namespace satisfies TF's later dlopen without any path search.

    import train.gpu  # noqa: must precede any tensorflow import
"""

import ctypes
import sys
from pathlib import Path

WSL_DRIVER = Path("/usr/lib/wsl/lib")
_loaded = False


def _candidates() -> list[Path]:
    root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia = root / "site-packages" / "nvidia"
    out = []
    if WSL_DRIVER.is_dir():
        out += sorted(WSL_DRIVER.glob("libcuda.so*"))
    if nvidia.is_dir():
        out += sorted(nvidia.glob("*/lib/*.so*"))
    return out


def preload() -> int:
    """Load every CUDA library found, dependencies first. -> count loaded."""
    global _loaded
    if _loaded:
        return 0
    libs = _candidates()
    n = 0
    # Several passes: a library may need one that has not been loaded yet, and
    # the dependency order is not knowable from the filenames alone.
    for _ in range(3):
        pending = []
        for p in libs:
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                n += 1
            except OSError:
                pending.append(p)
        libs = pending
        if not libs:
            break
    _loaded = True
    return n


preload()
