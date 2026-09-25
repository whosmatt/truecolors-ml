"""Sequence models over whole takes: causal dilated TCN, or a GRU.

The window MLP sees a fixed 2.9 s of pooled context. These see the raw block
stream and keep their own state, which is what the device would do anyway: a
TCN caches each layer's past activations, a GRU carries one state vector.

Both are causal and read `LOOKAHEAD` blocks past the label, like the MLP's two
future frames, so the output at block t is the prediction for t - LOOKAHEAD.
Scoring covers exactly the blocks the MLP scores: a full lookback inside the take.

    python -m train.seq --out runs/j_tcn32 --arch tcn --width 32
    python -m train.seq --out runs/j_gru64 --arch gru --width 64
    python -m train.seq --out runs/k_mil --arch tcn --melodic data/melodic

`--melodic` adds loops with an exact tempo and no trustworthy phase. Their beat
loss is phase-marginalised: the take is scored against the grid at every block
phase of its known period and only the best-fitting phase counts, so the model
chooses one phase per take and is taught the period, never a guessed phase.
"""

import argparse
import json
import time
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np
from keras import layers

from . import data, gridmetrics, mres, tempo
from . import model as model_mod

LOOKAHEAD = 2
BURN = mres.Spec().lookback  # the MLP's first scored centre


def takes(s):
    starts = np.flatnonzero(np.r_[True, s.take[1:] != s.take[:-1]])
    return np.split(np.arange(len(s.take)), starts[1:])


def pieces(groups, T):
    """Takes longer than T become overlapping rows; each row repeats BURN blocks
    of history, so every label is scored once and always with full context. One
    training take is 10,921 blocks against a p95 of 1,814: padded to it, every
    batch was 6x padding."""
    out = []
    for g in groups:
        s = 0
        while True:
            out.append(g[s:s + T])
            if s + T >= len(g):
                break
            s += T - BURN - LOOKAHEAD
    return out


def pack(s, mean, scale, targets, T=1344):
    """-> x (rows, T, F), y {head: (rows, T, w)}, idx (rows, T) block index or -1.

    Labels are shifted LOOKAHEAD blocks later so output t predicts label t - LOOKAHEAD,
    and masked (-1) before BURN, past the take end, and on padding.
    """
    groups = pieces(takes(s), T)
    T = max(len(g) for g in groups)
    Xn = ((s.X - mean) / scale).astype(np.float32)
    x = np.zeros((len(groups), T, Xn.shape[1]), np.float32)
    idx = np.full((len(groups), T), -1, np.int64)
    src = {"beat": s.beat,
           "beat_offset": np.where(s.beat > 0, s.beat_off, -1.0).astype(np.float32)}
    if "hit" in targets:
        src["hit"] = s.y
    if "beat_mil" in targets:
        src["beat_mil"] = np.where(s.beat < 0, s.period, -1.0).astype(np.float32)
    y = {k: np.full((len(groups), T, v.shape[1]), -1.0, np.float32) for k, v in src.items()}
    for i, g in enumerate(groups):
        n = len(g)
        x[i, :n] = Xn[g]
        lab = g[BURN:n - LOOKAHEAD]            # label blocks, as the MLP's centres
        t = np.arange(BURN, n - LOOKAHEAD) + LOOKAHEAD
        idx[i, t] = lab
        for k, v in src.items():
            y[k][i, t] = v[lab]
    return x, y, idx


PHASE_STEP = 1 / 3  # blocks; a coarser grid misplaces alternate beats of a fractional period
PHASES = 309        # covers the slowest period, 55 BPM = 102.3 blocks


def mil_bce(pos_weight):
    """y_true: the take's beat period in blocks on scored blocks, -1 elsewhere."""
    import tensorflow as tf

    def loss(y_true, y_pred):
        P = y_true[..., :1]                                         # (B, T, 1)
        m = tf.cast(P > 0, tf.float32)
        Pm = tf.maximum(P, 1.0)
        t = tf.cast(tf.range(tf.shape(P)[1]), tf.float32)[None, :, None]
        phi = tf.range(PHASES, dtype=tf.float32)[None, None, :] * PHASE_STEP  # (1, 1, K)
        # a beat at phi + n*P falls in block t when t <= phi + n*P < t + 1
        first = phi + tf.math.ceil((t - phi) / Pm) * Pm
        y = tf.cast(first < t + 1.0, tf.float32)                    # (B, T, K)
        # +-1 block tolerance: the phase grid, and the tempo-only label, cannot
        # place a beat to the block. Exact timing is left to the rendered clips.
        p = tf.clip_by_value(y_pred[..., :1], 1e-7, 1.0 - 1e-7)
        p_near = tf.nn.max_pool1d(p, 3, 1, "SAME")
        y_near = tf.nn.max_pool1d(y, 3, 1, "SAME")
        per = -(pos_weight * y * tf.math.log(p_near) + (1.0 - y_near) * tf.math.log(1.0 - p)) * m
        n = tf.reduce_sum(m, axis=1)                                # (B, 1)
        take_P = tf.reduce_max(P, axis=1)                           # (B, 1)
        cost = tf.reduce_sum(per, axis=1) / (n + 1e-6)              # (B, K)
        cost = tf.where(phi[:, 0] < take_P, cost, 1e9)              # phases inside one period
        valid = tf.cast(n[:, 0] > 0, tf.float32)
        return tf.reduce_sum(tf.reduce_min(cost, axis=1) * valid) / (tf.reduce_sum(valid) + 1e-6)

    loss.__name__ = "mil_bce"
    return loss


def tcn(n_feat, width, kernel, dilations, heads, dropout=0.1):
    inp = keras.Input(shape=(None, n_feat), name="features")
    h = layers.Conv1D(width, kernel, padding="causal", activation="relu", name="in")(inp)
    for d in dilations:
        r = layers.Conv1D(width, kernel, dilation_rate=d, padding="causal",
                          activation="relu", name=f"d{d}")(h)
        if dropout:
            r = layers.Dropout(dropout)(r)
        h = layers.Add()([h, r])
    return _heads(inp, h, heads, dropout)


def gru(n_feat, width, heads, dropout=0.1):
    inp = keras.Input(shape=(None, n_feat), name="features")
    h = layers.GRU(width, return_sequences=True, name="gru")(inp)
    return _heads(inp, h, heads, dropout)


def _heads(inp, h, heads, dropout):
    h = layers.Dense(64, activation="relu", name="trunk")(h)
    if dropout:
        h = layers.Dropout(dropout)(h)
    outs = {k: layers.Dense(w, activation=a, name=k)(h) for k, (w, a) in heads.items()}
    if "beat_mil" in heads:
        del outs["beat_mil"]
        outs["beat_mil"] = layers.Identity(name="beat_mil")(outs["beat"])
    return keras.Model(inp, outs)


def macs(m) -> int:
    """Per block, streaming: every weight is used once per new block."""
    total = 0
    for layer in m.layers:
        if isinstance(layer, (layers.Dense, layers.Conv1D)):
            total += int(np.prod(layer.kernel.shape))
        elif isinstance(layer, layers.GRU):
            total += sum(int(np.prod(w.shape)) for w in layer.weights if len(w.shape) == 2)
    return total


def activation(m, s, mean, scale, batch=64):
    """Dense per-block beat activation over a split, on the MLP's scored blocks."""
    x, _, idx = pack(s, mean, scale, ())
    a = np.zeros(len(s.X), np.float32)
    for i in range(0, len(x), batch):
        out = m.predict_on_batch(x[i:i + batch])["beat"][..., 0]
        k = idx[i:i + batch] >= 0
        a[idx[i:i + batch][k]] = np.asarray(out)[k]
    return a


def score(m, mean, scale, clips: Path, loops: Path) -> dict:
    te = data.load_many([clips], "test")
    out = {"clips": gridmetrics.per_take(activation(m, te, mean, scale), te)}
    lo = data.load_many([loops], "test")
    a = activation(m, lo, mean, scale)
    ex = oc = n = 0
    for g in takes(lo):
        true = 60.0 * tempo.BLOCK_HZ / float(np.median(lo.period[g][:, 0]))
        e1, e2 = tempo.tempo_ok(tempo.estimate(a[g]).bpm, true)
        ex, oc, n = ex + e1, oc + e2, n + 1
    out["loops"] = {"takes": n, "tempo": ex / n, "octave": oc / n}
    return out


class Batches(keras.utils.PyDataset):
    """Whole takes from host memory. The padded corpus is ~1 GB, too large for a
    GPU constant beside a concurrent run, and tf.data slicing it cost 200 ms/step
    against a 15 ms step."""

    def __init__(self, x, y, batch, seed=None):
        super().__init__()
        self.x, self.y, self.batch, self.seed = x, y, batch, seed
        self.order = np.arange(len(x))
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch))

    def __getitem__(self, i):
        k = np.sort(self.order[i * self.batch:(i + 1) * self.batch])
        return self.x[k], {h: v[k] for h, v in self.y.items()}

    def on_epoch_end(self):
        if self.seed is not None:
            np.random.default_rng(self.seed + self.epoch).shuffle(self.order)
            self.epoch += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=Path("data/features3"))
    ap.add_argument("--loops", type=Path, default=Path("data/loops"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", choices=("tcn", "gru"), required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16, help="takes per batch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--melodic", type=Path, default=None,
                    help="tempo-only loops, trained with the phase-marginalised beat loss")
    ap.add_argument("--mil-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)

    dirs = [a.clips] + ([a.melodic] if a.melodic else [])
    mean, scale = data.normaliser(data.load_many([a.clips], "train"))
    tr = data.load_many(dirs, "train")
    va = data.load_many(dirs, "val")
    if a.melodic:
        tr.period, va.period = (np.where(s_.beat < 0, s_.period, -1.0) for s_ in (tr, va))
    targets = ("beat", "beat_offset", "hit") + (("beat_mil",) if a.melodic else ())
    xt, yt, _ = pack(tr, mean, scale, targets)
    xv, yv, _ = pack(va, mean, scale, targets)

    heads = {"beat": (1, "sigmoid"), "beat_offset": (1, "sigmoid"), "hit": (tr.y.shape[1], "sigmoid")}
    pw = [a.pos_weight] * 3 + [1.0] * (tr.y.shape[1] - 3)
    losses = {"beat": model_mod.masked_bce(a.pos_weight), "beat_offset": model_mod.masked_mse,
              "hit": model_mod.masked_bce(pw)}
    weights = {"beat": 1.0, "beat_offset": 5.0, "hit": 0.3}
    if a.melodic:
        heads["beat_mil"] = (1, None)
        losses["beat_mil"] = mil_bce(a.pos_weight)
        weights["beat_mil"] = a.mil_weight
    F = tr.X.shape[1]
    m = (tcn(F, a.width, a.kernel, a.dilations, heads) if a.arch == "tcn"
         else gru(F, a.width, heads))
    rf = 1 + (a.kernel - 1) * (1 + sum(a.dilations)) if a.arch == "tcn" else None
    print(f"{a.arch} width {a.width} | receptive field {rf} | MACs/block {macs(m):,} | "
          f"params {m.count_params():,} | train takes {len(xt)}", flush=True)
    m.compile(optimizer=keras.optimizers.Adam(a.lr, clipnorm=1.0), loss=losses,
              loss_weights=weights, jit_compile=a.arch == "tcn")
    t0 = time.time()
    m.fit(Batches(xt, yt, a.batch, a.seed), validation_data=Batches(xv, yv, a.batch),
          epochs=a.epochs, verbose=2,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                   restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                       patience=2)])
    minutes = (time.time() - t0) / 60
    m.save(a.out / "model.keras")
    np.savez(a.out / "norm.npz", mean=mean, scale=scale)
    res = {"approach": "6-sequence", "arch": a.arch, "width": a.width, "kernel": a.kernel,
           "dilations": a.dilations if a.arch == "tcn" else None, "receptive_field": rf,
           "macs": macs(m), "params": m.count_params(), "seed": a.seed, "minutes": minutes,
           "lookahead": LOOKAHEAD, "melodic": str(a.melodic) if a.melodic else None,
           "mil_weight": a.mil_weight if a.melodic else None, "fe_spec_version": data.meta(a.clips)["fe_spec_version"],
           "test": score(m, mean, scale, a.clips, a.loops)}
    (a.out / "result.json").write_text(json.dumps(res, indent=2))
    c, l = res["test"]["clips"], res["test"]["loops"]
    print(f"usable {c['usable']:.3f} octave {c['octave']:.3f} phase {c['phase_exact']:.2f} ms | "
          f"loops tempo {l['tempo']:.3f} octave {l['octave']:.3f} | {minutes:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
