"""Approach 3: beat activation with multi-resolution context.

Same heads as approach 2, same stage 2. The only change is what stage 1 can see:
2.9 s of context instead of 171 ms, at three resolutions, which is what it takes
for "beat" and "offbeat" to be distinguishable at all.

    python -m train.grid3 --out runs/grid3
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
from .grid import build, grid_scores


def activation(model, split, mean, scale, spec, key="beat") -> np.ndarray:
    ds = mres.MResWindows(split, mean, scale, spec, targets=(), batch=8192)
    out = model.predict(ds, verbose=0)
    a = np.zeros(len(split.X), dtype=np.float32)
    v = out[key]
    a[ds.order] = v[:, 0] if v.shape[1] == 1 else np.maximum(v[:, 0], v[:, 1])
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("runs/grid3"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--mid", type=int, nargs=2, default=[16, 4])
    ap.add_argument("--coarse", type=int, nargs=2, default=[12, 16])
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)

    spec = mres.Spec(mid_frames=a.mid[0], mid_stride=a.mid[1],
                     coarse_frames=a.coarse[0], coarse_stride=a.coarse[1])
    meta = data.meta(a.features)
    tr = data.load(a.features, "train")
    va = data.load(a.features, "val")
    te = data.load(a.features, "test")
    if tr.beat is None:
        raise SystemExit("corpus has no beat labels; rebuild with dataset.build")
    mean, scale = data.normaliser(tr)
    targets = ("beat", "beat_offset") + (() if a.no_aux else ("hit",))

    m = build(spec.frames * len(meta["feature_order"]), tuple(a.hidden),
              not a.no_aux, n_aux=tr.y.shape[1])
    losses = {"beat": model_mod.weighted_bce(a.pos_weight),
              "beat_offset": model_mod.masked_mse}
    weights = {"beat": 1.0, "beat_offset": 5.0}
    if not a.no_aux:
        pw = [a.pos_weight] * 3 + [1.0] * (tr.y.shape[1] - 3)
        losses["hit"] = model_mod.weighted_bce(pw if tr.y.shape[1] > 3 else a.pos_weight)
        weights["hit"] = 0.3
    m.compile(optimizer=keras.optimizers.Adam(a.lr), loss=losses, loss_weights=weights)
    print(f"context {spec.seconds(evaluate.BLOCK_MS):.2f} s, {spec.frames} frames, "
          f"{spec.frames*len(meta['feature_order'])} inputs | MACs {model_mod.macs(m):,}")

    t0 = time.time()
    m.fit(
        mres.MResWindows(tr, mean, scale, spec, targets, batch=a.batch, shuffle=True),
        validation_data=mres.MResWindows(va, mean, scale, spec, targets, batch=a.batch),
        epochs=a.epochs, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                     patience=2)],
    )
    res = grid_scores(activation(m, te, mean, scale, spec), te)
    print(f"\ntrained in {(time.time()-t0)/60:.1f} min")
    print(f"\nTEST: rendered clips, exact grid ({res['takes']} takes)")
    for k in ("exact", "octave"):
        print(f"  {k:8} {res[k]:.1%}")
    print(f"  bpm err  {res['bpm_err']:.2f}%")
    print(f"  phase    median {res['phase_median']:.1f} ms   p90 {res['phase_p90']:.1f} ms")

    m.save(a.out / "model.keras")
    np.savez(a.out / "norm.npz", mean=mean, scale=scale)
    (a.out / "result.json").write_text(json.dumps({
        "approach": "3-multiresolution-context", "spec": spec.__dict__,
        "context_s": spec.seconds(evaluate.BLOCK_MS), "frames": spec.frames,
        "hidden": a.hidden, "aux": not a.no_aux, "macs": model_mod.macs(m),
        "seed": a.seed, "fe_variant": meta.get("fe_variant"),
        "fe_spec_version": meta["fe_spec_version"], "test_rendered": res,
    }, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
