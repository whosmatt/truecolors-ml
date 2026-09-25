"""GPU-resident training: the corpus, its pools and labels live on the device.

The host only feeds batch numbers. With the numpy PyDataset the GPU idled while
one CPU thread assembled each batch (11 ms, then 4 ms after pooling) for a model
that trains a batch in well under that.

    tr = fast.Corpus(split, mean, scale, spec, targets)
    t = fast.Trainer(net, tr, va, batch)
    t.compile(...); t.fit(t.train_data(), validation_data=t.val_data(), ...)
    net.save(...)   # the wrapped net, not the Trainer
"""

import keras
import numpy as np
import tensorflow as tf

from . import mres


class Corpus:
    """One split on the GPU: normalised frames, pools, per-head targets."""

    def __init__(self, split, mean, scale, spec: mres.Spec, targets):
        Xn = ((split.X - mean) / scale).astype(np.float32)
        views = spec.views()
        pools = [mres.pooled(Xn, st, k) for k, st, _ in views]
        offs = [o for _, _, o in views]
        # Offsets are gathered from one stacked array, so a window is a single
        # gather: row = pool_index * N + centre + offset.
        n = len(Xn)
        self.n = n
        self.frames = tf.constant(np.concatenate(pools))
        self.offsets = tf.constant(np.concatenate(
            [k * n + o for k, o in enumerate(offs)]).astype(np.int64))
        self.width = spec.width(Xn.shape[1])
        self.order = mres.valid_centres(split.take, spec)
        self.y = {k: tf.constant(v) for k, v in self._targets(split, targets).items()}

    @staticmethod
    def _targets(s, targets):
        out = {"beat": s.beat,
               "beat_offset": np.where(s.beat > 0, s.beat_off, -1.0).astype(np.float32)}
        if "hit" in targets:
            out["hit"] = s.y
        if "downbeat" in targets:
            out["downbeat"] = s.downbeat
            out["downbeat_offset"] = np.where(s.downbeat > 0, s.downbeat_off,
                                              -1.0).astype(np.float32)
        if "phrase" in targets:
            out["phrase"] = s.phrase
        if "music" in targets:
            out["music"] = s.music
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}

    def fetch(self, c):
        rows = tf.gather(self.frames, c[:, None] + self.offsets[None, :])
        return (tf.reshape(rows, (-1, self.width)),
                {k: tf.gather(v, c) for k, v in self.y.items()})


class Trainer(keras.Model):
    def __init__(self, net, train: Corpus, val: Corpus, batch: int, seed=0):
        super().__init__()
        self.net, self.tr, self.va, self.batch = net, train, val, batch
        self.perm = tf.Variable(self._padded(train.order), trainable=False)
        self.val_order = tf.constant(self._padded(val.order))
        self.lanes = tf.range(batch, dtype=tf.int64)
        self.seed = seed
        self.epoch = 0

    def call(self, x, training=False):
        return self.net(x, training=training)

    def _batch(self, corpus, order, i):
        # A gather, not a slice: XLA folds slice bounds into the program, so a
        # slice recompiled for every batch number (64 ms/step against 0.4).
        i = tf.reshape(tf.cast(i, tf.int64), ())
        return corpus.fetch(tf.gather(order, i * self.batch + self.lanes))

    def train_step(self, i):
        x, y = self._batch(self.tr, self.perm, i)
        with tf.GradientTape() as tape:
            yp = self.net(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=yp, training=True)
        # The default train_step feeds this tracker; overriding it silently
        # logs loss 0 and leaves EarlyStopping watching a constant.
        self._loss_tracker.update_state(loss, sample_weight=self.batch)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        return self.compute_metrics(x, y, yp)

    def test_step(self, i):
        x, y = self._batch(self.va, self.val_order, i)
        yp = self.net(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=yp, training=False)
        self._loss_tracker.update_state(loss, sample_weight=self.batch)
        return self.compute_metrics(x, y, yp)

    def _padded(self, order):
        """Wrap to a whole number of batches; repeats at most batch-1 centres."""
        return np.resize(order, self._steps_for(len(order)) * self.batch)

    def _steps_for(self, n):
        return int(np.ceil(n / self.batch))

    def reshuffle(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.perm.assign(self._padded(rng.permutation(self.tr.order)))
        self.epoch += 1

    def _steps(self, corpus):
        return self._steps_for(len(corpus.order))

    def train_data(self):
        return tf.data.Dataset.range(self._steps(self.tr)).prefetch(64)

    def val_data(self):
        return tf.data.Dataset.range(self._steps(self.va)).prefetch(64)


class Reshuffle(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reshuffle()


class Segments:
    """Tempo-only takes for the phase-marginalised beat loss: runs of `length`
    consecutive centres from one take, windowed like any Corpus."""

    def __init__(self, split, mean, scale, spec: mres.Spec, length: int):
        self.c = Corpus(split, mean, scale, spec, ())
        o = self.c.order
        # a start is usable when `length` consecutive valid centres share its take
        end = np.searchsorted(o, o + length - 1)
        ok = (end < len(o)) & (o[np.minimum(end, len(o) - 1)] == o + length - 1)
        ok &= split.take[np.minimum(o + length - 1, len(split.take) - 1)] == split.take[o]
        self.starts = tf.constant(o[ok])
        self.length = length
        self.period = tf.constant(split.period[:, 0].astype(np.float32))
        self.lanes = tf.range(length, dtype=tf.int64)

    def sample(self, n):
        k = tf.random.uniform((n,), 0, tf.shape(self.starts, out_type=tf.int64)[0], tf.int64)
        c = tf.gather(self.starts, k)[:, None] + self.lanes[None, :]     # (n, length)
        x, _ = self.c.fetch(tf.reshape(c, (-1,)))
        P = tf.gather(self.period, c)[..., None]                          # (n, length, 1)
        return x, P


class MilTrainer(Trainer):
    """Trainer plus a phase-marginalised beat loss on tempo-only segments."""

    def __init__(self, net, train, val, batch, segments: Segments, n_segments: int,
                 mil_loss, mil_weight: float, seed=0):
        super().__init__(net, train, val, batch, seed=seed)
        self.seg, self.n_seg, self.mil_loss, self.mil_weight = segments, n_segments, mil_loss, mil_weight
        self.mil_tracker = keras.metrics.Mean(name="mil")

    def train_step(self, i):
        x, y = self._batch(self.tr, self.perm, i)
        xs, P = self.seg.sample(self.n_seg)
        with tf.GradientTape() as tape:
            yp = self.net(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=yp, training=True)
            bs = self.net(xs, training=True)["beat"]
            mil = self.mil_loss(P, tf.reshape(bs, (self.n_seg, self.seg.length, 1)))
            total = loss + self.mil_weight * mil
        self._loss_tracker.update_state(loss, sample_weight=self.batch)
        self.mil_tracker.update_state(mil)
        grads = tape.gradient(total, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        out = self.compute_metrics(x, y, yp)
        out["mil"] = self.mil_tracker.result()
        return out
