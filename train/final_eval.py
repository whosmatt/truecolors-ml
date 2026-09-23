"""Every approach, on rendered clips and on real held-out drum loops.

Rendered clips carry an exact grid, so both tempo and phase are scored there.
Real loops carry a duration-verified tempo but no trustworthy phase, so only
tempo is scored on them — they are the only non-synthetic material available.

    python -m train.final_eval --loops 400
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from dataset import features as feat

from . import data, gridmetrics, mres, tempo
from .grid import activation as act_plain
from .grid3 import activation as act_mres

APPROACHES = [
    ("raw flux", None, None),
    ("1: drum detector", "runs/none", "hit"),
    ("2: beat activation", "runs/grid", "beat"),
    ("3: long context", "runs/grid3", "beat"),
]


def load(run):
    m = keras.models.load_model(Path(run) / "model.keras", compile=False)
    nz = np.load(Path(run) / "norm.npz")
    return m, nz["mean"], nz["scale"]


def act_for(name, run, key, X_split, spec):
    if run is None:
        return np.maximum(X_split.X[:, 5], X_split.X[:, 9])
    m, mean, scale = load(run)
    return (act_mres(m, X_split, mean, scale, spec, key=key) if name.startswith("3")
            else act_plain(m, X_split, mean, scale, key=key))


def loop_activation(name, run, key, X, spec):
    if run is None:
        return np.maximum(X[:, 5], X[:, 9])
    m, mean, scale = load(run)
    if name.startswith("3"):
        c = mres.valid_centres(np.zeros(len(X), dtype=np.int32), spec)
        cs = mres.cumsum(X)
        w = mres.gather(X, cs, c, spec).reshape(len(c), spec.frames, -1)
        xb = ((w - mean) / scale).reshape(len(c), -1).astype(np.float32)
    else:
        c = np.arange(data.PAST, len(X) - data.FUTURE)
        w = X[c[:, None] + np.arange(-data.PAST, data.FUTURE + 1)]
        xb = ((w - mean) / scale).reshape(len(c), -1).astype(np.float32)
    out = m.predict(xb, verbose=0, batch_size=8192)
    v = out[key]
    a = np.zeros(len(X), dtype=np.float32)
    a[c] = v[:, 0] if v.shape[1] == 1 else np.maximum(v[:, 0], v[:, 1])
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--loops", type=int, default=400)
    ap.add_argument("--notch", type=int, default=480)
    ap.add_argument("--out", type=Path, default=Path("results/3-long-context/final.json"))
    a = ap.parse_args()

    spec = mres.Spec()
    prior = tempo.training_prior(str(a.features))
    te = data.load(a.features, "test")
    rows = tempo.test_loops(a.manifest, a.loops)
    skip = int(tempo.SETTLE_S * tempo.BLOCK_HZ)

    results = {}
    for name, run, key in APPROACHES:
        r = {}
        act = act_for(name, run, key, te, spec)
        for tag, pr in (("", None), (" + prior", prior)):
            r["rendered" + tag] = _rendered(act, te, pr)
        # real loops: tempo only
        ex = {"": 0, " + prior": 0}
        oc = {"": 0, " + prior": 0}
        n = 0
        for row in rows:
            rng = np.random.default_rng(row["file_id"])
            pcm, _ = tempo.take_from_loop(row, rng, a.notch)
            if pcm is None:
                continue
            X = feat.featurise(pcm, hicut=True)
            if len(X) <= skip + spec.lookback + 200:
                continue
            aa = loop_activation(name, run, key, X, spec)[skip:]
            n += 1
            for tag, pr in (("", None), (" + prior", prior)):
                est = tempo.estimate(aa, prior=pr)
                e1, e2 = tempo.tempo_ok(est.bpm, row["bpm"])
                ex[tag] += e1
                oc[tag] += e2
        for tag in ("", " + prior"):
            r["loops" + tag] = {"n": n, "exact": ex[tag] / max(n, 1), "octave": oc[tag] / max(n, 1)}
        results[name] = r
        print(f"{name} done", flush=True)

    print(f"\n{'approach':20} {'rendered: usable':>17} {'+prior':>9} "
          f"{'loops: tempo':>13} {'+prior':>9} {'loops: +octave':>15} {'+prior':>9}")
    for name, _, _ in APPROACHES:
        r = results[name]
        print(f"{name:20} {r['rendered']['usable']:16.1%} {r['rendered + prior']['usable']:8.1%} "
              f"{r['loops']['exact']:12.1%} {r['loops + prior']['exact']:8.1%} "
              f"{r['loops']['octave']:14.1%} {r['loops + prior']['octave']:8.1%}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))


def _rendered(act, te, pr):
    import numpy as _np
    from . import evaluate
    usable = oc = n = 0
    for t in _np.unique(te.take):
        m = te.take == t
        idx = _np.flatnonzero(te.beat[m][:, 0] > 0)
        if len(idx) < 8:
            continue
        bt = (idx + te.beat_off[m][idx, 0]) * evaluate.BLOCK_MS
        true = 60000.0 / float(_np.median(_np.diff(bt)))
        n += 1
        est = tempo.estimate(act[m], prior=pr)
        _, e2 = tempo.tempo_ok(est.bpm, true)
        if not e2:
            continue
        oc += 1
        if tempo.grid_error_ms(est, true, float(bt[0] % (60000.0 / true))) <= gridmetrics.PHASE_OK_MS:
            usable += 1
    return {"takes": n, "usable": usable / max(n, 1), "octave": oc / max(n, 1)}


if __name__ == "__main__":
    main()
