"""Golden vector: one real window through the whole chain, for host verification.

Emits the raw front-end frames, the assembled 528-float window before and after
normalisation, the int8 input, and every model output — float and int8 — so a
firmware implementation can be checked stage by stage rather than end to end.

    python -m train.golden --out export/golden
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np
import tensorflow as tf

from . import data, export, mres

# Head order comes from the model, not a constant: a model with a music
# head has four outputs and hardcoding three silently drops one.
def head_order(model):
    return list(model.output_names) if hasattr(model, "output_names") \
        else list(model.output.keys())


def c_array(name: str, values, per_line=8, fmt="{:.8g}f") -> str:
    v = np.asarray(values).reshape(-1)
    body = []
    for i in range(0, len(v), per_line):
        body.append("    " + " ".join(fmt.format(x) + "," for x in v[i : i + per_line]))
    return f"static const float {name}[{len(v)}] = {{\n" + "\n".join(body) + "\n};\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/data3"))
    ap.add_argument("--features", type=Path, default=Path("data/features3"))
    ap.add_argument("--tflite", type=Path, default=Path("export/model.tflite"))
    ap.add_argument("--out", type=Path, default=Path("export/golden"))
    ap.add_argument("--blocks", type=int, default=420)
    ap.add_argument("--noise", type=Path, default=None,
                    help="add non_music and silence cases from this corpus; only "
                         "meaningful for a model with a music head")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    spec = mres.Spec()
    meta = data.meta(a.features)
    te = data.load(a.features, "test")
    nz = np.load(a.run / "norm.npz")
    mean, scale = nz["mean"], nz["scale"]

    # One take, and two centres from it: the strongest beat and the quietest
    # non-beat. A single near-silent centre verifies almost nothing, since the
    # model's job is to respond at beats.
    take = int(te.take[te.centres[0]])
    m = np.flatnonzero(te.take == take)
    frames = te.X[m][: a.blocks]
    csum = mres.cumsum(te.X)
    lo, hi = spec.lookback, len(frames) - spec.fine_future - 1
    cands = np.arange(lo, hi)
    model = keras.models.load_model(a.run / "model.keras", compile=False)
    w_all = mres.gather(te.X, csum, m[0] + cands, spec).reshape(len(cands), spec.frames, 12)
    n_all = ((w_all - mean) / scale).reshape(len(cands), -1).astype(np.float32)
    beats = np.asarray(model.predict(n_all, verbose=0)["beat"])[:, 0]

    blob = a.tflite.read_bytes()
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = sorted(it.get_output_details(), key=lambda d: int(d["name"].rsplit(":", 1)[1]))
    s, z = inp["quantization"]

    heads = head_order(model)

    def case(label, raw, norm, centre_local=None):
        f_out = model.predict(norm[None, :], verbose=0)
        q = np.clip(np.round(norm / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q[None, :])
        it.invoke()
        q_out = {}
        for name, d in zip(heads, outs):
            qs, qz = d["quantization"]
            q_out[name] = ((it.get_tensor(d["index"])[0].astype(np.float32) - qz) * qs).tolist()
        return {
            "label": label,
            "centre_in_frames": centre_local,
            "window_raw": [float(x) for x in raw],
            "window_normalised": [float(x) for x in norm],
            "input_int8": [int(x) for x in q],
            "output_float": {kk: [float(x) for x in np.asarray(v)[0]] for kk, v in f_out.items()},
            "output_int8_dequantised": q_out,
        }

    cases = [case(label, w_all[k].reshape(-1), n_all[k], int(cands[k]))
             for label, k in (("beat", int(np.argmax(beats))), ("no_beat", int(np.argmin(beats))))]

    # Both cases above come from one music take, so a music head read from the
    # wrong tensor or inverted still passes them. These two cannot.
    if a.noise is not None:
        nzs = data.load(a.noise, "test")
        c = mres.valid_centres(nzs.take, spec)
        c = np.random.default_rng(0).choice(c, min(20000, len(c)), replace=False)
        w = mres.gather(nzs.X, mres.cumsum(nzs.X), c, spec).reshape(len(c), spec.frames, 12)
        n = ((w - mean) / scale).reshape(len(c), -1).astype(np.float32)
        music = np.asarray(model.predict(n, verbose=0, batch_size=4096)["music"])[:, 0]
        spl = w[:, :, 11].mean(1)
        loud = spl > 70.0
        k_nm = int(np.flatnonzero(loud)[np.argmin(music[loud])])
        k_si = int(np.argmin(spl))
        cases += [case("non_music", w[k_nm].reshape(-1), n[k_nm]),
                  case("silence", w[k_si].reshape(-1), n[k_si])]

    doc = {
        "note": "frames[i] is one fe_out_t in feature_order; each case names the "
                "centre block index into frames",
        "feature_order": meta["feature_order"],
        "fe_spec_version": meta["fe_spec_version"],
        "fe_variant": meta.get("fe_variant"),
        "output_order": heads,
        "context": {**export.context_spec(spec.__dict__), "frames": spec.frames,
                    "inputs": spec.frames * 12, "lookback_blocks": spec.lookback,
                    "ring_blocks": spec.lookback + 1 + spec.fine_future},
        "frame_layout": [
            "fine: 16 frames, block t-13 .. t+2, OLDEST to NEWEST",
            "mid: 16 frames, means of 4 blocks, NEWEST to OLDEST, "
            "frame k covers blocks [t-17-4k, t-13-4k)",
            "coarse: 12 frames, means of 16 blocks, NEWEST to OLDEST, "
            "frame k covers blocks [t-93-16k, t-77-16k)",
        ],
        "normalisation": {"mean": [float(x) for x in mean],
                          "scale": [float(x) for x in scale]},
        "input_quant": {"scale": float(s), "zero_point": int(z)},
        "output_quant": {n: {"scale": float(d["quantization"][0]),
                             "zero_point": int(d["quantization"][1])}
                         for n, d in zip(heads, outs)},
        "frames": [[float(x) for x in row] for row in frames],
        "window_only_cases": "cases with centre_in_frames null come from other takes: "
                             "test them from window_raw, not from frames",
        "cases": cases,
    }
    (a.out / "golden.json").write_text(json.dumps(doc, indent=1))

    h = ["// Generated by `python -m train.golden`. Golden vector for the",
         "// approach-3 model: raw front-end frames -> window -> model outputs.",
         "// Outputs are concatenated in signature order: " + ", ".join(heads) + ".",
         "// Frame layout: fine 16 (t-13..t+2, oldest first), then mid 16 (means",
         "// of 4 blocks, newest first), then coarse 12 (means of 16, newest first).",
         "#pragma once", "",
         f"#define GOLDEN_BLOCKS {len(frames)}",
         f"#define GOLDEN_FRAMES {spec.frames}",
         f"#define GOLDEN_INPUTS {spec.frames * 12}",
         f"#define GOLDEN_RING_BLOCKS {spec.lookback + 1 + spec.fine_future}", "",
         c_array("golden_fe_frames", frames, per_line=12),
         c_array("golden_norm_mean", mean, per_line=6),
         c_array("golden_norm_scale", scale, per_line=6),
         f"static const float golden_input_scale = {float(s):.10g}f;",
         f"static const int   golden_input_zero  = {int(z)};", ""]
    for c in cases:
        p = c["label"]
        if c["centre_in_frames"] is not None:
            h.append(f"#define GOLDEN_{p.upper()}_CENTRE {c['centre_in_frames']}")
        h += [
              c_array(f"golden_{p}_window_raw", c["window_raw"]),
              c_array(f"golden_{p}_window_norm", c["window_normalised"]),
              c_array(f"golden_{p}_out_float",
                      [v for h in heads for v in np.atleast_1d(c["output_float"][h])]),
              c_array(f"golden_{p}_out_int8",
                      [v for h in heads for v in np.atleast_1d(c["output_int8_dequantised"][h])])]
    (a.out / "golden.h").write_text("\n".join(h))

    print(f"take {take}, {len(frames)} frames")
    for c in cases:
        o, q = c["output_float"], c["output_int8_dequantised"]
        print(f"  {c['label']:9} centre {str(c['centre_in_frames']):>4}  " + "  ".join(
            f"{h_} {o[h_][0]:.4f}/{q[h_][0]:.4f}" for h_ in heads if h_ != "hit"))
    for p in sorted(a.out.iterdir()):
        print(f"  {p.name:14} {p.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
