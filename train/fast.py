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
        pools = (Xn, mres.pooled(Xn, spec.mid_stride), mres.pooled(Xn, spec.coarse_stride))
        mid0 = -spec.fine_past
        crs0 = mid0 - spec.mid_frames * spec.mid_stride
        offs = (np.arange(-spec.fine_past, spec.fine_future + 1),
                mid0 - np.arange(spec.mid_frames) * spec.mid_stride,
                crs0 - np.arange(spec.coarse_frames) * spec.coarse_stride)
        # Offsets are gathered from one stacked array, so a window is a single
        # gather: row = pool_index * N + centre + offset.
        n = len(Xn)
        self.n = n
        self.frames = tf.constant(np.concatenate(pools))
        self.offsets = tf.constant(np.concatenate(
            [k * n + o for k, o in enumerate(offs)]).astype(np.int64))
        self.width = spec.frames * Xn.shape[1]
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
