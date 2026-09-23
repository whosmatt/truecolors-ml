"""One trainer, heads selectable, so each change is a single flag and a single
scoreboard row.

    python -m train.grid5 --out runs/e_base                    # matches data3
    python -m train.grid5 --out runs/e_music   --music
    python -m train.grid5 --out runs/e_down    --downbeat --phrase
    python -m train.grid5 --out runs/e_noaux   --no-aux

Every head trains under a mask, so corpora that lack a label simply do not
contribute to it: the non-music takes have no grid, the loops have no beats.
"""

import argparse
import json
import time
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, fast, mres
from . import model as model_mod


SILENT_DB = 52.0     # front-end floor is 50 dB; 6.1% of rendered-clip blocks
SILENT_BLOCKS = 64   # and 0% of real loops sit below this (val, 2026-09-23)


def silent(s) -> np.ndarray:
    """Mean spl_db over the past 683 ms, per block, below SILENT_DB."""
    spl = s.X[:, -1].astype(np.float64)
    c = np.r_[0.0, np.cumsum(spl)]
    r = np.full(len(spl), np.inf)
    r[SILENT_BLOCKS:] = (c[SILENT_BLOCKS + 1:] - c[1:-SILENT_BLOCKS]) / SILENT_BLOCKS
    return r < SILENT_DB


def build(n_input, hidden, heads: dict, dropout=0.1) -> keras.Model:
    inp = keras.Input(shape=(n_input,), name="features")
    x = inp
    for i, h in enumerate(hidden):
        x = keras.layers.Dense(h, activation="relu", name=f"dense{i}")(x)
        if dropout:
            x = keras.layers.Dropout(dropout, name=f"drop{i}")(x)
    outs = {}
    for name, (width, act) in heads.items():
        outs[name] = keras.layers.Dense(width, activation=act, name=name)(x)
    return keras.Model(inp, outs, name="grid5")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=Path("data/features3"))
    ap.add_argument("--noise", type=Path, default=Path("data/noise"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--no-aux", action="store_true", help="drop the drum-class head")
    ap.add_argument("--downbeat", action="store_true")
    ap.add_argument("--phrase", action="store_true")
    ap.add_argument("--music", action="store_true", help="adds the non-music corpus")
    ap.add_argument("--music-loops", action="store_true",
                    help="real drum loops as music positives (grid stays masked)")
    ap.add_argument("--music-silence", action="store_true",
                    help="blocks whose past 683 ms sit at the floor are not music")
    ap.add_argument("--music-pos", type=float, default=1.0,
                    help="positive weight for the music head; loops tip it ~7:1 positive")
    ap.add_argument("--loops", type=Path, default=Path("data/loops"))
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--downbeat-weight", type=float, default=1.0,
                    help="loss weight for the downbeat head")
    ap.add_argument("--downbeat-pos", type=float, default=None,
                    help="positive weight for downbeats; default 4x --pos-weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-per-execution", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)

    spec = mres.Spec()
    meta = data.meta(a.clips)
    dirs = [a.clips] + ([a.noise] if a.music else [])
    fill = [{}] * len(dirs)
    if a.music_loops:
        dirs.append(a.loops)
        fill.append({"music": 1.0})
    tr = data.load_many(dirs, "train", fill=fill)
    va = data.load_many(dirs, "val", fill=fill)
    if a.music_silence:
        for s_ in (tr, va):
            s_.music[silent(s_) & (s_.music[:, 0] >= 0)] = 0.0
    mean, scale = data.normaliser(tr)

    heads = {"beat": (1, "sigmoid"), "beat_offset": (1, "sigmoid")}
    targets = ["beat", "beat_offset"]
    losses = {"beat": model_mod.masked_bce(a.pos_weight),
              "beat_offset": model_mod.masked_mse}
    weights = {"beat": 1.0, "beat_offset": 5.0}
    if not a.no_aux:
        heads["hit"] = (tr.y.shape[1], "sigmoid")
        targets.append("hit")
        pw = [a.pos_weight] * 3 + [1.0] * (tr.y.shape[1] - 3)
        losses["hit"] = model_mod.masked_bce(pw)
        weights["hit"] = 0.3
    if a.downbeat:
        heads["downbeat"] = (1, "sigmoid")
        heads["downbeat_offset"] = (1, "sigmoid")
        targets.append("downbeat")
        # Downbeats are a quarter as frequent as beats. The 4x default was a
        # guess and measured badly; it is a flag so it can be tuned rather than
        # assumed.
        losses["downbeat"] = model_mod.masked_bce(a.downbeat_pos or a.pos_weight * 4)
        losses["downbeat_offset"] = model_mod.masked_mse
        weights.update({"downbeat": a.downbeat_weight,
                        "downbeat_offset": a.downbeat_weight * 2})
    if a.phrase:
        heads["phrase"] = (1, None)
        targets.append("phrase")
        losses["phrase"] = model_mod.masked_log_mse
        weights["phrase"] = 1.0
    if a.music:
        heads["music"] = (1, "sigmoid")
        targets.append("music")
        losses["music"] = model_mod.masked_bce(a.music_pos)
        weights["music"] = 1.0

    m = build(spec.frames * len(meta["feature_order"]), tuple(a.hidden), heads)
    print(f"heads {list(heads)} | train {len(tr.X):,} blocks | MACs {model_mod.macs(m):,}")

    t0 = time.time()
    trainer = fast.Trainer(m, fast.Corpus(tr, mean, scale, spec, targets),
                           fast.Corpus(va, mean, scale, spec, targets), a.batch, seed=a.seed)
    trainer.compile(optimizer=keras.optimizers.Adam(a.lr), loss=losses,
                    loss_weights=weights, steps_per_execution=a.steps_per_execution,
                    jit_compile=True)
    trainer.fit(
        trainer.train_data(), validation_data=trainer.val_data(),
        epochs=a.epochs, verbose=2,
        callbacks=[fast.Reshuffle(),
                   keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                     patience=2)],
    )
    m.save(a.out / "model.keras")
    np.savez(a.out / "norm.npz", mean=mean, scale=scale)
    (a.out / "result.json").write_text(json.dumps({
        "approach": "5-configurable", "heads": {k: v[0] for k, v in heads.items()},
        "spec": spec.__dict__, "hidden": a.hidden, "macs": model_mod.macs(m),
        "seed": a.seed, "aux": not a.no_aux, "downbeat": a.downbeat,
        "phrase": a.phrase, "music": a.music, "music_loops": a.music_loops,
        "music_silence": a.music_silence, "music_pos": a.music_pos,
        "train_blocks": int(len(tr.X)), "minutes": (time.time() - t0) / 60,
        "fe_variant": meta.get("fe_variant"),
        "fe_spec_version": meta["fe_spec_version"],
    }, indent=2))
    print(f"trained in {(time.time()-t0)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
