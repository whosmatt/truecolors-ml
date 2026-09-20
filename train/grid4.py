"""Approach 4: joint training on rendered clips and real loops.

The rendered clips carry an exact grid; the 8,381 BPM-verified drum loops carry
an exact *period* and no trustworthy phase. Masked losses let each contribute
what it knows: beats and drums from the clips, period from both. That adds
20 hours of real audio to a corpus otherwise made of 985 synthesised grooves.

The period head also gives stage 2 a per-take tempo estimate, which is a far
sharper constraint than the global "music is usually near 120 BPM" prior.

    python -m train.grid4 --out runs/grid4
"""

import argparse
import json
import time
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, evaluate, mres
from . import model as model_mod


def build(n_input: int, hidden, aux_width: int, dropout: float = 0.1) -> keras.Model:
    inp = keras.Input(shape=(n_input,), name="features")
    x = inp
    for i, h in enumerate(hidden):
        x = keras.layers.Dense(h, activation="relu", name=f"dense{i}")(x)
        if dropout:
            x = keras.layers.Dropout(dropout, name=f"drop{i}")(x)
    return keras.Model(inp, {
        "beat": keras.layers.Dense(1, activation="sigmoid", name="beat")(x),
        "beat_offset": keras.layers.Dense(1, activation="sigmoid", name="beat_offset")(x),
        "hit": keras.layers.Dense(aux_width, activation="sigmoid", name="hit")(x),
        # log2 of the beat period in blocks; linear, and the only head the loop
        # takes can train
        "period": keras.layers.Dense(1, name="period")(x),
    }, name="grid4")


class Windows(mres.MResWindows):
    def __getitem__(self, i):
        xb, out = super().__getitem__(i)
        c = self.order[i * self.batch : (i + 1) * self.batch]
        out["period"] = self.s.period[c]
        return xb, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=Path("data/features3"))
    ap.add_argument("--loops", type=Path, default=Path("data/loops"))
    ap.add_argument("--out", type=Path, default=Path("runs/grid4"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--no-loops", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)

    dirs = [a.clips] if a.no_loops else [a.clips, a.loops]
    spec = mres.Spec()
    meta = data.meta(a.clips)
    tr = data.load_many(dirs, "train")
    va = data.load_many(dirs, "val")
    mean, scale = data.normaliser(tr)
    targets = ("beat", "beat_offset", "hit")
    n_real = int((tr.beat[:, 0] >= 0).sum())
    print(f"train {len(tr.X):,} blocks ({n_real/len(tr.X):.0%} with a grid, "
          f"{1-n_real/len(tr.X):.0%} period-only)")

    m = build(spec.frames * len(meta["feature_order"]), tuple(a.hidden), tr.y.shape[1])
    pw = [a.pos_weight] * 3 + [1.0] * (tr.y.shape[1] - 3)
    m.compile(
        optimizer=keras.optimizers.Adam(a.lr),
        loss={"beat": model_mod.masked_bce(a.pos_weight),
              "beat_offset": model_mod.masked_mse,
              "hit": model_mod.masked_bce(pw),
              "period": model_mod.masked_log_mse},
        loss_weights={"beat": 1.0, "beat_offset": 5.0, "hit": 0.3, "period": 2.0},
    )
    print(f"context {spec.seconds(evaluate.BLOCK_MS):.2f} s | MACs {model_mod.macs(m):,}")

    t0 = time.time()
    m.fit(
        Windows(tr, mean, scale, spec, targets, batch=a.batch, shuffle=True),
        validation_data=Windows(va, mean, scale, spec, targets, batch=a.batch),
        epochs=a.epochs, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                     patience=2)],
    )
    m.save(a.out / "model.keras")
    np.savez(a.out / "norm.npz", mean=mean, scale=scale)
    (a.out / "result.json").write_text(json.dumps({
        "approach": "4-joint-loops", "spec": spec.__dict__,
        "context_s": spec.seconds(evaluate.BLOCK_MS), "frames": spec.frames,
        "hidden": a.hidden, "macs": model_mod.macs(m), "seed": a.seed,
        "loops": not a.no_loops, "clips": str(a.clips),
        "train_blocks": int(len(tr.X)), "grid_fraction": n_real / len(tr.X),
        "minutes": (time.time() - t0) / 60,
        "fe_variant": meta.get("fe_variant"), "fe_spec_version": meta["fe_spec_version"],
    }, indent=2))
    print(f"\ntrained in {(time.time()-t0)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
