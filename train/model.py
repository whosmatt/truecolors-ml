"""Stage 1: per-block onset classification, with an optional sub-block head.

Dense, not convolutional: on this silicon esp-nn's fully-connected kernel runs at
0.95 cyc/MAC against 11-26 for every conv1d shape measured, and per-call overhead
favours few large layers over many small ones. See agents/compute-budget.md.
"""

import keras
import numpy as np
from keras import layers, ops

POS_WEIGHT = 20.0  # positives are ~1.2% of blocks


def weighted_bce(pos_weight: float = POS_WEIGHT):
    def loss(y_true, y_pred):
        p = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        return -ops.mean(
            pos_weight * y_true * ops.log(p) + (1.0 - y_true) * ops.log(1.0 - p)
        )

    loss.__name__ = "weighted_bce"
    return loss


def masked_mse(y_true, y_pred):
    """Sub-block position, scored only where an onset actually is."""
    m = ops.cast(y_true >= 0.0, "float32")
    return ops.sum(m * ops.square(y_pred - y_true)) / (ops.sum(m) + 1e-6)


def build(
    n_input: int,
    n_classes: int = 3,
    hidden: tuple[int, ...] = (128, 64),
    offset_head: bool = True,
    dropout: float = 0.1,
) -> keras.Model:
    inp = keras.Input(shape=(n_input,), name="features")
    x = inp
    for i, h in enumerate(hidden):
        x = layers.Dense(h, activation="relu", name=f"dense{i}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"drop{i}")(x)
    outs = {"hit": layers.Dense(n_classes, activation="sigmoid", name="hit")(x)}
    if offset_head:
        outs["offset"] = layers.Dense(n_classes, activation="sigmoid", name="offset")(x)
    return keras.Model(inp, outs, name="onset")


def macs(model: keras.Model) -> int:
    """Multiply-accumulates per inference. The budget is ~150k."""
    total = 0
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            total += int(np.prod(layer.kernel.shape))
    return total
