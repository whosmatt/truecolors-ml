"""Approach 2: predict the beat grid directly.

Stage 1 stops classifying drums and predicts "a beat falls in this block"
instead — which is what stage 2 actually consumes. Stage 2 is unchanged, so the
comparison against approach 1 isolates what the activation is worth.

Supervision comes from the rendered .alc clips, whose MIDI puts beat 0 at the
clip start, so the grid is exact. Deriving it from real loops was measured at
20 ms median error with 17% off-beat locks — see agents/model-plan.md.

    python -m train.grid --out runs/grid
"""

import argparse
import json
import time
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, evaluate, tempo
from . import model as model_mod


def build(n_input: int, hidden, aux: bool, dropout: float = 0.1, n_aux: int = 3) -> keras.Model:
    inp = keras.Input(shape=(n_input,), name="features")
    x = inp
    for i, h in enumerate(hidden):
        x = keras.layers.Dense(h, activation="relu", name=f"dense{i}")(x)
        if dropout:
            x = keras.layers.Dropout(dropout, name=f"drop{i}")(x)
    outs = {
        "beat": keras.layers.Dense(1, activation="sigmoid", name="beat")(x),
        "beat_offset": keras.layers.Dense(1, activation="sigmoid", name="beat_offset")(x),
    }
    if aux:
        # Drum classes as an auxiliary task: free supervision from the same
        # takes, and a beat is usually a drum hit, so the features overlap.
        outs["hit"] = keras.layers.Dense(n_aux, activation="sigmoid", name="hit")(x)
    return keras.Model(inp, outs, name="grid")


def activation(model, split, mean, scale, key="beat") -> np.ndarray:
    ds = data.Windows(split, mean, scale, batch=8192, shuffle=False, targets=())
    out = model.predict(ds, verbose=0)
    a = np.zeros(len(split.X), dtype=np.float32)
    v = out[key]
    a[split.centres] = v[:, 0] if v.shape[1] == 1 else np.maximum(v[:, 0], v[:, 1])
    return a


def grid_scores(act: np.ndarray, split: data.Split, label="") -> dict:
    """Tempo and phase per take, against the exact rendered grid."""
    ex = oc = n = 0
    ph, be = [], []
    for t in np.unique(split.take):
        m = split.take == t
        idx = np.flatnonzero(split.beat[m][:, 0] > 0)
        if len(idx) < 8:
            continue
        bt = (idx + split.beat_off[m][idx, 0]) * evaluate.BLOCK_MS
        true_bpm = 60000.0 / float(np.median(np.diff(bt)))
        est = tempo.estimate(act[m])
        e1, e2 = tempo.tempo_ok(est.bpm, true_bpm)
        n += 1
        ex += e1
        oc += e2
        if np.isfinite(est.bpm):
            r = min((1.0, 2.0, 0.5, 3.0, 1 / 3, 1.5, 2 / 3),
                    key=lambda k: abs(est.bpm * k - true_bpm))
            be.append(abs(est.bpm * r - true_bpm) / true_bpm * 100)
        if e2:
            ph.append(tempo.grid_error_ms(est, true_bpm, float(bt[0] % (60000.0 / true_bpm))))
    ph = np.array(ph) if ph else np.array([np.nan])
    return {"takes": n, "exact": ex / max(n, 1), "octave": oc / max(n, 1),
            "bpm_err": float(np.median(be)) if be else float("nan"),
            "phase_median": float(np.median(ph)), "phase_p90": float(np.percentile(ph, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("runs/grid"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--past", type=int, default=data.PAST)
    ap.add_argument("--future", type=int, default=data.FUTURE)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(a.seed)
    data.PAST, data.FUTURE = a.past, a.future
    data.WINDOW = a.past + a.future + 1

    meta = data.meta(a.features)
    tr = data.load(a.features, "train", a.past, a.future)
    va = data.load(a.features, "val", a.past, a.future)
    te = data.load(a.features, "test", a.past, a.future)
    if tr.beat is None:
        raise SystemExit("corpus has no beat labels; rebuild with dataset.build")
    mean, scale = data.normaliser(tr)
    targets = ("beat", "beat_offset") + (() if a.no_aux else ("hit",))

    m = build(data.WINDOW * len(meta["feature_order"]), tuple(a.hidden),
              not a.no_aux, n_aux=tr.y.shape[1])
    losses = {"beat": model_mod.weighted_bce(a.pos_weight),
              "beat_offset": model_mod.masked_mse}
    weights = {"beat": 1.0, "beat_offset": 5.0}
    if not a.no_aux:
        pw = [a.pos_weight] * 3 + [1.0] * (tr.y.shape[1] - 3)
        losses["hit"] = model_mod.weighted_bce(pw if tr.y.shape[1] > 3 else a.pos_weight)
        weights["hit"] = 0.3
    m.compile(optimizer=keras.optimizers.Adam(a.lr), loss=losses, loss_weights=weights)
    print(f"window {data.WINDOW} blocks ({data.WINDOW*evaluate.BLOCK_MS:.0f} ms) | "
          f"MACs {model_mod.macs(m):,} | aux {not a.no_aux}")

    t0 = time.time()
    m.fit(
        data.Windows(tr, mean, scale, batch=a.batch, shuffle=True, targets=targets,
                     past=a.past, future=a.future),
        validation_data=data.Windows(va, mean, scale, batch=a.batch, targets=targets,
                                     past=a.past, future=a.future),
        epochs=a.epochs, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                     patience=2)],
    )
    res = grid_scores(activation(m, te, mean, scale), te)
    print(f"\ntrained in {(time.time()-t0)/60:.1f} min")
    print(f"\nTEST — rendered clips, exact grid ({res['takes']} takes)")
    print(f"  tempo within 4%      {res['exact']:.1%}")
    print(f"  +octave              {res['octave']:.1%}")
    print(f"  bpm err (octave-fold) {res['bpm_err']:.2f}%")
    print(f"  phase median         {res['phase_median']:.1f} ms")
    print(f"  phase p90            {res['phase_p90']:.1f} ms")

    m.save(a.out / "model.keras")
    np.savez(a.out / "norm.npz", mean=mean, scale=scale)
    (a.out / "result.json").write_text(json.dumps({
        "approach": "2-beat-activation", "window": data.WINDOW,
        "past": a.past, "future": a.future, "hidden": a.hidden,
        "aux": not a.no_aux, "macs": model_mod.macs(m), "seed": a.seed,
        "fe_variant": meta.get("fe_variant"), "fe_spec_version": meta["fe_spec_version"],
        "test_rendered": res,
    }, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
