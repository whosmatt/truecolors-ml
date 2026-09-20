"""Train stage 1 and evaluate it in milliseconds.

    python -m train.run --out runs/base
    python -m train.run --out runs/no-offset --no-offset-head
"""

import argparse
import json
import time
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from dataset.ableton import DETECTION_CLASSES

from . import data, evaluate, model as model_mod

# Which raw feature a naive detector would use per class, for the baseline.
BASELINE_FEATURE = {"kick": "flux0", "snare": "mid_flux", "hihat": "treble_flux"}


def predict(m, split: data.Split, mean, scale, batch=8192) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Dense per-block predictions, aligned back onto the block index."""
    ds = data.Windows(split, mean, scale, batch=batch, shuffle=False)
    out = m.predict(ds, verbose=0)
    # Size from the model, not the labels: a model trained without the none
    # class has fewer outputs than the label array has columns.
    hit = np.zeros((len(split.X), out["hit"].shape[1]), dtype=np.float32)
    off = (np.zeros((len(split.X), out["offset"].shape[1]), dtype=np.float32)
           if "offset" in out else None)
    hit[split.centres] = out["hit"]
    if off is not None:
        off[split.centres] = out["offset"]
    return hit, off, split.centres


def baseline(split: data.Split, feature_order: list[str]) -> dict:
    """A naive per-feature threshold, to put the model's numbers in context."""
    scores = {}
    for ci, cls in enumerate(DETECTION_CLASSES):
        col = feature_order.index(BASELINE_FEATURE[cls])
        sig = split.X[:, col].astype(np.float32)
        sig = sig / (sig.max() or 1.0)
        th, s = evaluate.best_threshold(
            sig, None, split.y[:, ci], split.off[:, ci], split.take
        )
        scores[cls] = s
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("runs/base"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--no-offset-head", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=model_mod.POS_WEIGHT)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)

    meta = data.meta(a.features)
    tr = data.load(a.features, "train")
    va = data.load(a.features, "val")
    te = data.load(a.features, "test")
    mean, scale = data.normaliser(tr)
    print(f"train {len(tr):,} windows | val {len(va):,} | test {len(te):,}")
    print(f"window {data.WINDOW} blocks ({data.WINDOW * evaluate.BLOCK_MS:.0f} ms), "
          f"lookahead {data.FUTURE} blocks ({data.FUTURE * evaluate.BLOCK_MS:.1f} ms)")

    m = model_mod.build(
        data.WINDOW * len(meta["feature_order"]),
        n_classes=tr.y.shape[1],
        n_offsets=tr.off.shape[1],
        hidden=tuple(a.hidden),
        offset_head=not a.no_offset_head,
    )
    # none is the majority class and needs no up-weighting
    pw = [a.pos_weight] * len(DETECTION_CLASSES) + [1.0] * (tr.y.shape[1] - len(DETECTION_CLASSES))
    losses = {"hit": model_mod.weighted_bce(pw if tr.y.shape[1] > len(DETECTION_CLASSES)
                                            else a.pos_weight)}
    if not a.no_offset_head:
        losses["offset"] = model_mod.masked_mse
    m.compile(optimizer=keras.optimizers.Adam(a.lr), loss=losses,
              loss_weights={"hit": 1.0, **({"offset": 5.0} if not a.no_offset_head else {})})
    print(f"MACs {model_mod.macs(m):,} | int8 weights ~{model_mod.macs(m)/1024:.0f} KB")

    t0 = time.time()
    use_off = not a.no_offset_head
    m.fit(
        data.Windows(tr, mean, scale, batch=a.batch, shuffle=True, with_offset=use_off),
        validation_data=data.Windows(va, mean, scale, batch=a.batch, with_offset=use_off),
        epochs=a.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        ],
        verbose=2,
    )
    train_s = time.time() - t0

    # Thresholds are chosen on val and applied unchanged to test.
    hv, ov, _ = predict(m, va, mean, scale)
    ht, ot, _ = predict(m, te, mean, scale)
    thresholds, val_scores, test_scores, test_nooff = {}, {}, {}, {}
    for ci, cls in enumerate(DETECTION_CLASSES):
        th, s = evaluate.best_threshold(hv[:, ci], None if ov is None else ov[:, ci],
                                        va.y[:, ci], va.off[:, ci], va.take)
        thresholds[cls] = th
        val_scores[cls] = s
        test_scores[cls] = evaluate.score_class(
            ht[:, ci], None if ot is None else ot[:, ci], te.y[:, ci], te.off[:, ci], te.take, th)
        # same model, onsets placed at block centre: isolates what the head buys
        test_nooff[cls] = evaluate.score_class(
            ht[:, ci], None, te.y[:, ci], te.off[:, ci], te.take, th)

    print(f"\ntrained in {train_s/60:.1f} min | thresholds "
          + ", ".join(f"{c} {t:.2f}" for c, t in thresholds.items()))
    evaluate.report("TEST — model", test_scores)
    if ot is not None:
        evaluate.report("TEST — same model, onset at block centre (no offset head)", test_nooff)
    if not a.skip_baseline:
        evaluate.report("TEST — naive single-feature threshold", baseline(te, meta["feature_order"]))

    m.save(a.out / "model.keras")
    np.savez(a.out / "norm.npz", mean=mean, scale=scale)
    (a.out / "result.json").write_text(json.dumps({
        "window": data.WINDOW, "past": data.PAST, "future": data.FUTURE,
        "hidden": a.hidden, "offset_head": not a.no_offset_head,
        "pos_weight": a.pos_weight, "macs": model_mod.macs(m), "seed": a.seed,
        "thresholds": thresholds, "train_minutes": train_s / 60,
        "fe_spec_version": meta["fe_spec_version"],
        "dataset": {"manifest_sha256": meta["manifest_sha256"], "ir_sha256": meta.get("ir_sha256")},
        "val": {k: v.asdict() for k, v in val_scores.items()},
        "test": {k: v.asdict() for k, v in test_scores.items()},
        "test_block_centre": {k: v.asdict() for k, v in test_nooff.items()},
    }, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
