"""Multi-resolution context windows.

Approaches 1 and 2 both see 171 ms, which is less than half a beat. Nothing in
that window distinguishes a beat from an offbeat, which is why their phase error
sits near half a beat for a fifth of takes no matter what the head predicts.

Feeding seconds of context at full rate is unaffordable: 3 s is 280 blocks, and
280x12 into a 128-unit layer is 430k MACs against a ~150k budget. So context is
kept at the resolution it is needed at, which is how the ear works too: the last
few blocks exactly, the last beat coarsely, the last few bars coarser still.

    fine   13 past + 2 future, full rate     171 ms
    mid    16 frames of 4 blocks             683 ms
    coarse 12 frames of 16 blocks            2048 ms
                                             2.9 s total, 44 frames, 528 inputs
"""

from dataclasses import dataclass

import keras
import numpy as np


@dataclass(frozen=True)
class Spec:
    fine_past: int = 13
    fine_future: int = 2
    mid_frames: int = 16
    mid_stride: int = 4
    coarse_frames: int = 12
    coarse_stride: int = 16

    @property
    def lookback(self) -> int:
        return (self.fine_past + self.mid_frames * self.mid_stride
                + self.coarse_frames * self.coarse_stride)

    @property
    def frames(self) -> int:
        return self.fine_past + self.fine_future + 1 + self.mid_frames + self.coarse_frames

    def seconds(self, block_ms: float) -> float:
        return self.lookback * block_ms / 1000.0


def valid_centres(take: np.ndarray, spec: Spec) -> np.ndarray:
    """Centres whose whole context lies inside one take."""
    n = len(take)
    idx = np.arange(n)
    ok = (idx >= spec.lookback) & (idx < n - spec.fine_future)
    lo, hi = spec.lookback, n - spec.fine_future
    ok[lo:hi] &= take[: n - spec.lookback - spec.fine_future] == take[lo + spec.fine_future:]
    return np.flatnonzero(ok).astype(np.int64)


def gather(X: np.ndarray, csum: np.ndarray, centres: np.ndarray, spec: Spec) -> np.ndarray:
    """-> (len(centres), frames * n_features), fine to coarse."""
    parts = [X[centres[:, None] + np.arange(-spec.fine_past, spec.fine_future + 1)]]
    base = centres - spec.fine_past
    for nf, st in ((spec.mid_frames, spec.mid_stride),
                   (spec.coarse_frames, spec.coarse_stride)):
        ends = base[:, None] - np.arange(nf)[None, :] * st
        starts = ends - st
        # csum has a leading zero row, so mean(a:b) = (csum[b] - csum[a]) / (b - a)
        parts.append((csum[ends] - csum[starts]) / st)
        base = base - nf * st
    return np.concatenate(parts, axis=1).reshape(len(centres), -1)


def cumsum(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1, X.shape[1]), dtype=np.float64),
                           np.cumsum(X, axis=0, dtype=np.float64)], axis=0)


def pooled(X: np.ndarray, stride: int) -> np.ndarray:
    """P[j] = mean(X[j-stride:j]); rows below stride are zero and never gathered."""
    c = cumsum(X)
    out = np.zeros(X.shape, dtype=np.float32)
    out[stride:] = (c[stride:-1] - c[:-stride - 1]) / stride
    return out


class MResWindows(keras.utils.PyDataset):
    """Same windows as gather(), assembled from precomputed pools.

    Normalising before pooling is exact (the normaliser is affine per feature),
    so batches reduce to three row gathers. The cumsum path cost 11 ms/batch,
    46 s of each ~54 s epoch.
    """

    def __init__(self, split, mean, scale, spec: Spec, targets, batch=1024,
                 shuffle=False, seed=0, **kw):
        super().__init__(**kw)
        self.s, self.spec, self.targets, self.batch = split, spec, targets, batch
        Xn = ((split.X - mean) / scale).astype(np.float32)
        self.pools = (Xn, pooled(Xn, spec.mid_stride), pooled(Xn, spec.coarse_stride))
        mid0 = -spec.fine_past
        crs0 = mid0 - spec.mid_frames * spec.mid_stride
        self.offsets = (np.arange(-spec.fine_past, spec.fine_future + 1),
                        mid0 - np.arange(spec.mid_frames) * spec.mid_stride,
                        crs0 - np.arange(spec.coarse_frames) * spec.coarse_stride)
        self.order = valid_centres(split.take, spec)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.order)

    def window(self, c):
        return np.concatenate([P[c[:, None] + o] for P, o in zip(self.pools, self.offsets)],
                              axis=1).reshape(len(c), -1)

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch))

    def __getitem__(self, i):
        c = self.order[i * self.batch : (i + 1) * self.batch]
        xb = self.window(c)
        out = {}
        if "beat" in self.targets:
            out["beat"] = self.s.beat[c]
        if "beat_offset" in self.targets:
            out["beat_offset"] = np.where(self.s.beat[c] > 0, self.s.beat_off[c],
                                          -1.0).astype(np.float32)
        if "hit" in self.targets:
            out["hit"] = self.s.y[c]
        return xb, out

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)
