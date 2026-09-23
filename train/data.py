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
    downbeat: np.ndarray | None = None      # (N, 1) bar starts; -1 masked
    downbeat_off: np.ndarray | None = None  # (N, 1)
    phrase: np.ndarray | None = None        # (N, 1) loop length in beats
    music: np.ndarray | None = None         # (N, 1) 1 music, 0 not, -1 masked

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
                 d["period"] if "period" in d.files else None,
                 *(d[k] if k in d.files else None
                   for k in ("downbeat", "downbeat_off", "phrase", "music")))


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
    out = np.full((len(s.X), 1), -1.0, dtype=np.float32)
    # Takes are contiguous runs; a full-length mask per take was O(N * takes)
    # and cost 180 s on features3.
    starts = np.flatnonzero(np.r_[True, s.take[1:] != s.take[:-1]])
    ends = np.r_[starts[1:], len(s.take)]
    for a, b in zip(starts, ends):
        idx = np.flatnonzero(s.beat[a:b, 0] > 0)
        if len(idx) < 4:
            continue
        pos = idx + s.beat_off[a:b][idx, 0]
        out[a:b] = float(np.median(np.diff(pos)))
    return out


def load_many(dirs, split: str, past: int = PAST, future: int = FUTURE,
              fill=None) -> Split:
    """Merge corpora, keeping take ids unique.

    Corpora without a period column get one derived from their beat labels;
    corpora without beat labels carry -1 and are masked out of those losses.
    `fill`, one dict per dir, supplies constant labels a corpus implies.
    """
    parts, base = [], 0
    for d, f in zip(dirs, fill or [{}] * len(dirs)):
        s = load(Path(d), split, past, future)
        # Constant labels a corpus lacks but implies, e.g. every loop is music.
        for k, v in f.items():
            if getattr(s, k) is None:
                setattr(s, k, np.full((len(s.X), 1), v, dtype=np.float32))
        if s.period is None:
            s.period = period_from_beats(s) if s.beat is not None else np.full(
                (len(s.X), 1), -1.0, dtype=np.float32)
        s.take = s.take + base
        base = int(s.take.max()) + 1
        parts.append(s)
    if len(parts) == 1:
        return parts[0]
    def cat(f, width=1):
        cols = []
        for p in parts:
            v = getattr(p, f)
            cols.append(np.full((len(p.X), width), -1.0, dtype=np.float32) if v is None else v)
        return np.concatenate(cols)

    merged = Split(cat("X", parts[0].X.shape[1]), cat("y", parts[0].y.shape[1]),
                   cat("off", parts[0].off.shape[1]),
                   np.concatenate([p.take for p in parts]),
                   np.zeros(0, dtype=np.int64), cat("beat"), cat("beat_off"),
                   cat("period"), cat("downbeat"), cat("downbeat_off"),
                   cat("phrase"), cat("music"))
    # recompute valid centres over the merged take ids
    n = len(merged.X)
    idx = np.arange(n)
    ok = (idx >= past) & (idx < n - future)
    ok[past : n - future] &= merged.take[: n - past - future] == merged.take[past + future :]
    merged.centres = np.flatnonzero(ok).astype(np.int64)
    return merged
