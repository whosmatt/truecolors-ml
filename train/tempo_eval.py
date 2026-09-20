"""Compare what stage 2 is fed: discrete onsets, model activation, or raw flux.

    python -m train.tempo_eval --run runs/nocomb --n 300
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from dataset import features as feat
from dataset.ableton import DETECTION_CLASSES

from . import data, evaluate, tempo


def activations(X, model, mean, scale):
    c = np.arange(data.PAST, len(X) - data.FUTURE)
    w = X[c[:, None] + np.arange(-data.PAST, data.FUTURE + 1)]
    out = model.predict(((w - mean) / scale).reshape(len(c), -1), verbose=0, batch_size=8192)
    key = "beat" if "beat" in out else "hit"
    hit = np.zeros((len(X), out[key].shape[1]), dtype=np.float32)
    hit[c] = out[key]
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/nocomb"))
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--notch", type=int, default=480)
    ap.add_argument("--no-comb", action="store_true", default=True)
    a = ap.parse_args()

    model = keras.models.load_model(a.run / "model.keras", compile=False)
    nz = np.load(a.run / "norm.npz")
    mean, scale = nz["mean"], nz["scale"]
    th = json.loads((a.run / "result.json").read_text())["thresholds"]

    rows = tempo.test_loops(a.manifest, a.n)
    skip = int(tempo.SETTLE_S * tempo.BLOCK_HZ)
    methods = ("activation", "onsets", "raw flux")
    acc1 = {m: 0 for m in methods}
    acc2 = {m: 0 for m in methods}
    phase = {m: [] for m in methods}
    bpm_err = {m: [] for m in methods}
    n = 0

    for i, row in enumerate(rows):
        rng = np.random.default_rng(row["file_id"])
        pcm, true_phase = tempo.take_from_loop(row, rng, a.notch)
        if pcm is None:
            continue
        X = feat.featurise(pcm, a.notch, comb=not a.no_comb, hicut=True)
        if len(X) <= skip + 400:
            continue
        hit = activations(X, model, mean, scale)
        n += 1
        # The activation is trimmed by the AGC settle window, so the true grid
        # phase has to be re-referenced to that same origin.
        true_phase_at_skip = (true_phase - skip * tempo.BLOCK_MS) % (60000.0 / row["bpm"])

        # kick and snare carry the beat; hats subdivide it
        act = np.maximum(hit[:, 0], hit[:, 1])[skip:]
        onset = np.zeros_like(act)
        for ci, cls in enumerate(("kick", "snare")):
            for p in evaluate.pick_peaks(hit[skip:, ci], th[cls]):
                onset[p] = max(onset[p], 1.0)
        flux = np.maximum(X[skip:, 5], X[skip:, 9])  # flux0 and mid_flux

        for m, sig in (("activation", act), ("onsets", onset), ("raw flux", flux)):
            est = tempo.estimate(sig)
            e1, e2 = tempo.tempo_ok(est.bpm, row["bpm"])
            acc1[m] += e1
            acc2[m] += e2
            if np.isfinite(est.bpm):
                # Fold octave errors out: for a metronome a half/double-time grid
                # still lands on real beats, so report the error after folding.
                r = min((1.0, 2.0, 0.5, 3.0, 1 / 3, 1.5, 2 / 3),
                        key=lambda k: abs(est.bpm * k - row["bpm"]))
                bpm_err[m].append(abs(est.bpm * r - row["bpm"]) / row["bpm"] * 100)
            if e2:
                phase[m].append(tempo.grid_error_ms(est, row["bpm"], true_phase_at_skip))

    print(f"\n{n} real drum loops, none seen in training, 20 s each "
          f"through IR + coil whine\n")
    print(f"{'stage 2 input':14} {'tempo ok':>9} {'+octave':>9} {'bpm err %':>10} "
          f"{'phase ms':>10} {'phase p90':>10}")
    for m in methods:
        pe = np.array(phase[m]) if phase[m] else np.array([np.nan])
        be = np.array(bpm_err[m]) if bpm_err[m] else np.array([np.nan])
        print(f"{m:14} {acc1[m]/n:9.1%} {acc2[m]/n:9.1%} {np.median(be):10.2f} "
              f"{np.median(pe):10.1f} {np.percentile(pe,90):10.1f}")


if __name__ == "__main__":
    main()
