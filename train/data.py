"""Windowed views over the block-feature corpus.

A window never crosses a take boundary: takes have independent AGC state and
different kits, so a window spanning two of them is a signal that cannot occur on
the device.

Windows are gathered per batch rather than materialised. The full corpus at
16x12 per block would be 2.2 GB; the (N, 12) array plus an index is 137 MB.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np

PAST = 13    # blocks of history
FUTURE = 2   # blocks of lookahead: energy from an onset spreads into the next
             # block or two (measured +0.5 to +1.3), and a fixed latency is
             # exactly compensable by the beatgrid stage downstream
WINDOW = PAST + FUTURE + 1


@dataclass
class Split:
    X: np.ndarray       # (N, 12) block features
    y: np.ndarray       # (N, C) onset in this block
    off: np.ndarray     # (N, C) position within the block, [0, 1)
    take: np.ndarray    # (N,) take id
    centres: np.ndarray # indices with a full window inside one take
    beat: np.ndarray | None = None      # (N, 1) beat grid, exact; -1 = masked
    beat_off: np.ndarray | None = None  # (N, 1) sub-block position
    period: np.ndarray | None = None    # (N, 1) beat period in blocks

    def __len__(self) -> int:
        return len(self.centres)


def load(path: Path, split: str, past: int = PAST, future: int = FUTURE) -> Split:
    d = np.load(Path(path) / f"{split}.npz")
    X, y, off, take = d["X"], d["y"], d["off"], d["take"]
    # A centre is valid when the whole window shares its take id.
    n = len(X)
    idx = np.arange(n)
    ok = (idx >= past) & (idx < n - future)
    ok[past : n - future] &= take[: n - past - future] == take[past + future :]
    return Split(X, y, off, take, np.flatnonzero(ok).astype(np.int64),
                 d["beat"] if "beat" in d.files else None,
                 d["beat_off"] if "beat_off" in d.files else None,
                 d["period"] if "period" in d.files else None)


def normaliser(s: Split) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/scale from the training split.

    Fixed at export time and burned into the firmware's input path, so it is
    computed once on train and reused everywhere — never per split.
    """
    mean = s.X.mean(axis=0)
    scale = s.X.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class Windows(keras.utils.PyDataset):
    def __init__(
        self,
        s: Split,
        mean: np.ndarray,
        scale: np.ndarray,
        batch: int = 1024,
        shuffle: bool = False,
        past: int = PAST,
        future: int = FUTURE,
        seed: int = 0,
        with_offset: bool = True,
        targets: tuple[str, ...] = ("hit", "offset"),
        **kw,
    ):
        super().__init__(**kw)
        self.s, self.mean, self.scale, self.batch = s, mean, scale, batch
        self.with_offset = with_offset
        self.targets = targets
        self.offsets = np.arange(-past, future + 1)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = s.centres.copy()
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return int(np.ceil(len(self.order) / self.batch))

    def __getitem__(self, i):
        c = self.order[i * self.batch : (i + 1) * self.batch]
        w = self.s.X[c[:, None] + self.offsets]          # (B, W, 12)
        w = (w - self.mean) / self.scale
        xb = w.reshape(len(c), -1)
        hit = self.s.y[c]
        # -1 marks "no onset here", so the offset loss can mask itself: there is
        # no meaningful sub-block position for a block with no onset in it.
        out = {}
        if "hit" in self.targets:
            out["hit"] = hit
        if "offset" in self.targets:
            k = self.s.off.shape[1]
            out["offset"] = np.where(hit[:, :k] > 0, self.s.off[c], -1.0).astype(np.float32)
        if "beat" in self.targets:
            out["beat"] = self.s.beat[c]
        if "beat_offset" in self.targets:
            out["beat_offset"] = np.where(
                self.s.beat[c] > 0, self.s.beat_off[c], -1.0
            ).astype(np.float32)
        return xb, out

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def meta(path: Path) -> dict:
    return json.loads((Path(path) / "meta.json").read_text())


def period_from_beats(s: Split) -> np.ndarray:
    """Per-take beat period in blocks, read off exact beat labels."""
    out = np.zeros((len(s.X), 1), dtype=np.float32)
    for t in np.unique(s.take):
        m = s.take == t
        idx = np.flatnonzero(s.beat[m][:, 0] > 0)
        if len(idx) < 4:
            out[m] = -1.0
            continue
        pos = idx + s.beat_off[m][idx, 0]
        out[m] = float(np.median(np.diff(pos)))
    return out


def load_many(dirs, split: str, past: int = PAST, future: int = FUTURE) -> Split:
    """Merge corpora, keeping take ids unique.

    Corpora without a period column get one derived from their beat labels;
    corpora without beat labels carry -1 and are masked out of those losses.
    """
    parts, base = [], 0
    for d in dirs:
        s = load(Path(d), split, past, future)
        if s.period is None:
            s.period = period_from_beats(s) if s.beat is not None else np.full(
                (len(s.X), 1), -1.0, dtype=np.float32)
        s.take = s.take + base
        base = int(s.take.max()) + 1
        parts.append(s)
    if len(parts) == 1:
        return parts[0]
    cat = lambda f: np.concatenate([getattr(p, f) for p in parts])
    merged = Split(cat("X"), cat("y"), cat("off"), cat("take"),
                   np.zeros(0, dtype=np.int64), cat("beat"), cat("beat_off"),
                   cat("period"))
    # recompute valid centres over the merged take ids
    n = len(merged.X)
    idx = np.arange(n)
    ok = (idx >= past) & (idx < n - future)
    ok[past : n - future] &= merged.take[: n - past - future] == merged.take[past + future :]
    merged.centres = np.flatnonzero(ok).astype(np.int64)
    return merged
