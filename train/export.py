"""Quantise the model to int8 TFLite and verify

    python -m train.export --run runs/data3 --out export/
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np
import tensorflow as tf

from . import data, evaluate, gridmetrics, mres, tempo
from .grid3 import activation as act_float


def representative(split, mean, scale, spec, n=500, batch=64, seed=0):
    ds = mres.MResWindows(split, mean, scale, spec, targets=(), batch=batch,
                          shuffle=True, seed=seed)

    def gen():
        for i in range(min(n, len(ds))):
            x, _ = ds[i]
            for row in x:
                yield [row[None, :].astype(np.float32)]

    return gen


def to_tflite(model, rep) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    # Full int8: esp-nn has no float kernels, and a float fallback op would run
    # in TFLM's reference implementation at a large cost.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(blob: bytes, split, mean, scale, spec, names):
    """Dense int8 predictions, aligned back onto the block index.

    Outputs are matched by their position in the signature, not by name: the
    converter renames them to `StatefulPartitionedCall_1:N`, and guessing from
    those strings silently mixed up the beat and offset heads.
    """
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = sorted(it.get_output_details(), key=lambda d: int(d["name"].rsplit(":", 1)[1]))
    assert len(outs) == len(names), f"expected {names}, got {len(outs)} outputs"
    in_scale, in_zero = inp["quantization"]

    ds = mres.MResWindows(split, mean, scale, spec, targets=(), batch=1024)
    acc = {n: np.zeros(len(split.X), dtype=np.float32) for n in names
           if n in ("beat", "beat_offset", "music")}
    pos = 0
    for i in range(len(ds)):
        x, _ = ds[i]
        q = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
        for row in q:
            it.set_tensor(inp["index"], row[None, :])
            it.invoke()
            for name, d in zip(names, outs):
                if name not in acc:
                    continue
                s, z = d["quantization"]
                acc[name][ds.order[pos]] = (float(it.get_tensor(d["index"])[0, 0]) - z) * s
            pos += 1
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/data3"))
    ap.add_argument("--features", type=Path, default=Path("data/features3"))
    ap.add_argument("--out", type=Path, default=Path("export"))
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--calib-extra", type=Path, nargs="*", default=[],
                    help="more corpora for int8 calibration, e.g. data/noise data/loops "
                         "for a music head that must see them in range")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    spec = mres.Spec()
    meta = data.meta(a.features)
    tr = data.load(a.features, "train")
    te = data.load(a.features, "test")
    nz = np.load(a.run / "norm.npz")
    mean, scale = nz["mean"], nz["scale"]
    model = keras.models.load_model(a.run / "model.keras", compile=False)

    print("converting...", flush=True)
    t0 = time.time()
    calib = data.load_many([a.features, *a.calib_extra], "train") if a.calib_extra else tr
    blob = to_tflite(model, representative(calib, mean, scale, spec, n=a.calib))
    (a.out / "model.tflite").write_bytes(blob)
    print(f"  {len(blob)/1024:.1f} KB in {time.time()-t0:.0f}s")

    print("scoring float...", flush=True)
    af = act_float(model, te, mean, scale, spec, key="beat")
    rf = gridmetrics.per_take(af, te)
    print("scoring int8...", flush=True)
    heads = list(model.output_names) if hasattr(model, "output_names") else list(model.output.keys())
    q = run_tflite(blob, te, mean, scale, spec, heads)
    aq = q["beat"]
    rq = gridmetrics.per_take(aq, te)

    corr = float(np.corrcoef(af[te.centres], aq[te.centres])[0, 1])
    print(f"\n{'':10} {'usable':>8} {'octave':>8} {'phase exact':>12}")
    for lbl, r in (("float32", rf), ("int8", rq)):
        print(f"{lbl:10} {r['usable']:8.1%} {r['octave']:8.1%} {r['phase_exact']:11.2f}ms")
    print(f"\nactivation correlation float vs int8: {corr:.4f}")

    run = json.loads((a.run / "result.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    model_meta = {
        "fe_spec_version": meta["fe_spec_version"],
        "fe_variant": meta.get("fe_variant"),
        "fe_variant_name": meta.get("fe_variant_name"),
        "feature_order": meta["feature_order"],
        "block_samples": meta["block_samples"],
        "sample_rate": 48000,
        "context": {**run["spec"], "frames": spec.frames,
                    "inputs": spec.frames * len(meta["feature_order"]),
                    "seconds": spec.seconds(evaluate.BLOCK_MS),
                    "lookback_blocks": spec.lookback,
                    "ring_blocks": spec.lookback + 1 + spec.fine_future,
                    "lookahead_blocks": spec.fine_future},
        "normalisation": {"mean": [float(x) for x in mean],
                          "scale": [float(x) for x in scale]},
        "outputs": {k: int(v.shape[-1]) for k, v in zip(
            heads, (model.outputs if isinstance(model.outputs, list) else [model.outputs]))},
        "output_order": heads,
        "hit_classes": ["kick", "snare", "hihat", "none"],
        "macs": run["macs"],
        "tflite_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "trained_from": {"repo_commit": commit,
                         "manifest_sha256": meta["manifest_sha256"],
                         "ir_sha256": meta.get("ir_sha256"),
                         "corpus": str(a.features), "run": str(a.run)},
        "metrics": {"float32": rf, "int8": rq,
                    "activation_correlation": corr},
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (a.out / "model_meta.json").write_text(json.dumps(model_meta, indent=2))
    print(f"\n-> {a.out}/model.tflite, model_meta.json")


if __name__ == "__main__":
    main()
