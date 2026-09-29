"""A model fed its own past beat activation, run block by block.

The feedback column fb3 is the activation shifted by 3 blocks: the newest
context frame is block t+2 and the newest own output at centre t is t-1. A run
trained with `--aux fb3_<A>` saw model A's activation there (open loop); here it
sees its own (closed loop).

The first layer is linear up to its ReLU, so its pre-activation splits into a
part from the audio features, computed once in batch with the feedback column
at 0, and a part from the feedback frames, G[f] (128 per frame, folded through
the per-frame projection when there is one) times each frame's value. Stepping
is then 44 x 128 plus the small trunk per take and block.

    python -m train.closedloop runs/e_fb_s0 data/features4_mel ...   # writes aux_fb3_<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import mres

SHIFT = 3


def fb_series(act: np.ndarray, take: np.ndarray) -> np.ndarray:
    """act -> the fb3 column: act[j - SHIFT] within a take, 0 before."""
    fb = np.zeros_like(act)
    fb[SHIFT:] = act[:-SHIFT]
    starts = np.flatnonzero(np.r_[True, take[1:] != take[:-1]])
    for s in starts:
        fb[s: s + SHIFT] = 0.0
    return fb


class Loop:
    def __init__(self, run: Path):
        from . import gpu  # noqa: F401  must precede keras
        import keras

        from . import scoreboard
        self.run = Path(run)
        self.info = json.loads((self.run / "result.json").read_text())
        self.m = keras.models.load_model(self.run / "model.keras", compile=False)
        nz = np.load(self.run / "norm.npz")
        self.mean, self.scale = nz["mean"], nz["scale"]
        self.spec = scoreboard.run_spec(self.run)
        cascade = self.info.get("cascade") or []
        assert cascade and cascade[-1].startswith("fb3_"), "not a self-feedback run"
        self.n_feat = self.info["n_feat"]
        self.fb_col = self.n_feat - 1
        W0 = self.m.get_layer("dense0").kernel.numpy()
        self.b0 = self.m.get_layer("dense0").bias.numpy()
        n_frames = self.spec.width(1)
        if self.info.get("proj"):
            Kp = self.m.get_layer("proj").kernel.numpy()           # (n_feat, P)
            P = Kp.shape[1]
            self.G = np.stack([Kp[self.fb_col] @ W0[f * P:(f + 1) * P] for f in range(n_frames)])
        else:
            self.G = np.stack([W0[f * self.n_feat + self.fb_col] for f in range(n_frames)])
        self.G = (self.G / self.scale[self.fb_col]).astype(np.float32)  # per raw fb unit
        self.W1 = self.m.get_layer("dense1").kernel.numpy()
        self.b1 = self.m.get_layer("dense1").bias.numpy()
        self.Wb = self.m.get_layer("beat").kernel.numpy()[:, 0]
        self.bb = float(self.m.get_layer("beat").bias.numpy()[0])
        self.pre_model = keras.Model(self.m.input, self.m.get_layer("dense0").input)

    def features_pre(self, s) -> tuple[np.ndarray, np.ndarray]:
        """-> (valid centres, dense0 pre-activation there with fb = 0), s.X without fb."""
        from dataclasses import replace
        X = np.concatenate([s.X, np.zeros((len(s.X), 1), np.float32)], axis=1)
        ds = mres.MResWindows(replace(s, X=X), self.mean, self.scale, self.spec,
                              targets=(), batch=8192)
        W0 = self.m.get_layer("dense0").kernel.numpy()
        pre = np.empty((len(ds.order), W0.shape[1]), np.float16)
        for i in range(len(ds)):
            xb = ds[i][0]
            h = self.pre_model.predict_on_batch(xb) if self.info.get("proj") else xb
            pre[i * 8192: i * 8192 + len(xb)] = np.asarray(h) @ W0 + self.b0
        return ds.order, pre

    def run_split(self, s, feedback: np.ndarray | None = None) -> np.ndarray:
        """Beat activation over a split whose X lacks the fb column. With
        `feedback` (an activation series), open loop on it instead; for tests."""
        order, pre = self.features_pre(s)
        n = len(s.X)
        act = np.zeros(n, np.float32)
        slot = np.full(n, -1, np.int64)
        slot[order] = np.arange(len(order))
        views = self.spec.views()
        fine_off = views[0][2]
        pooled = [(st, off) for _, st, off in views[1:]]
        starts = np.flatnonzero(np.r_[True, s.take[1:] != s.take[:-1]])
        bounds = list(zip(starts, np.r_[starts[1:], n]))
        bounds.sort(key=lambda b: b[1] - b[0])
        for g in range(0, len(bounds), 512):
            grp = bounds[g: g + 512]
            L = max(e - b for b, e in grp)
            B = len(grp)
            fb = np.zeros((B, L + SHIFT + 1), np.float32)
            C = np.zeros((B, L + SHIFT + 2), np.float64)  # C[:, j] = fb[:, :j].sum()
            idx = np.full((B, L), -1, np.int64)
            ext = None
            if feedback is not None:
                ext = np.zeros((B, L), np.float32)
            for k, (b, e) in enumerate(grp):
                idx[k, : e - b] = slot[b:e]
                if ext is not None:
                    ext[k, : e - b] = feedback[b:e]
            out = np.zeros((B, L), np.float32)
            for t in range(L):
                live = idx[:, t] >= 0
                if live.any():
                    frames = [fb[:, t + fine_off]]
                    for st, off in pooled:
                        j = t + off
                        frames.append(((C[:, j] - C[:, j - st]) / st).astype(np.float32))
                    v = np.concatenate(frames, axis=1)                 # (B, n_frames)
                    h = pre[np.maximum(idx[:, t], 0)].astype(np.float32) + v @ self.G
                    h = np.maximum(h, 0.0)
                    h = np.maximum(h @ self.W1 + self.b1, 0.0)
                    y = 1.0 / (1.0 + np.exp(-(h @ self.Wb + self.bb)))
                    out[:, t] = np.where(live, y, 0.0)
                src = ext[:, t] if ext is not None else out[:, t]
                fb[:, t + SHIFT] = src
                C[:, t + 1] = C[:, t] + fb[:, t]
            for k, (b, e) in enumerate(grp):
                act[b:e] = out[k, : e - b]
        return act


def activation(run, s) -> np.ndarray:
    """Closed-loop beat activation for an evaluation split (X as the run sees it, fb dropped)."""
    from dataclasses import replace
    return Loop(run).run_split(replace(s, X=s.X[:, :-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    from . import data
    loop = Loop(a.run)
    tag = f"fb3_{Path(a.run).name}"
    base_aux = [t for t in loop.info["cascade"][:-1]]
    for c in a.corpora:
        for sp in a.splits:
            if not (Path(c) / f"{sp}.npz").exists():
                continue
            s = data.load(Path(c), sp, mel=loop.info.get("mel"), aux=base_aux)
            act = loop.run_split(s)
            np.save(Path(c) / f"aux_{tag}_{sp}.npy", fb_series(act, s.take)[:, None])
            print(f"{c} {sp}: {len(act):,} blocks", flush=True)


if __name__ == "__main__":
    main()
