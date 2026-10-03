"""Export a two-stage model (stage A: mel + FE; stage B: the same plus A's
shifted beat activation) for the firmware, verify float against int8 end to end,
and write golden vectors from real audio.

Each stage splits at its per-frame projection. The projection is linear and the
context tiers are means, so the device projects every block once in float
(normalisation folded in) and pools projected values; the int8 TFLite trunk then
takes the same 44 x 12 = 528-input layout as the single-stage models.

    python -m train.export2 --a runs/final_d0_s0 --b runs/final_g_s0 --out export/twostage1
"""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np
import tensorflow as tf

from dataset import features
from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

from . import closedloop, compare, data, evaluate, gridmetrics, mres, scoreboard
from .export import context_spec, to_tflite

CALIB = ("data/features4_fe3", "data/noise_fe3", "data/loops_grid_fe3", "data/melodic_fe3",
         "data/songs_all_fe3", "data/playlists_all_fe3")
TEST = {"clips": "data/features4_fe3", "loops": "data/loops_fe3", "melodic": "data/melodic_fe3",
        "loops_grid": "data/loops_grid_fe3", "melodic_grid": "data/melodic_grid_fe3"}
HIT_CLASSES = ["kick", "snare", "hihat", "none"]


class Stage:
    def __init__(self, run: Path, heads=None):
        self.run = run
        self.info = json.loads((run / "result.json").read_text())
        self.m = keras.models.load_model(run / "model.keras", compile=False)
        nz = np.load(run / "norm.npz")
        self.mean, self.scale = nz["mean"].astype(np.float64), nz["scale"].astype(np.float64)
        self.spec = scoreboard.run_spec(run)
        assert self.info.get("proj") and self.info.get("mel") == "flux16"
        self.heads = heads or list(self.info["heads"])
        self.pre = keras.Model(self.m.input, self.m.get_layer("projected").output)
        inp = keras.Input((self.pre.output.shape[-1],), name="projected")
        x = self.m.get_layer("dense0")(inp)
        i = 1
        while True:
            try:
                x = self.m.get_layer(f"dense{i}")(x)
                i += 1
            except ValueError:
                break
        self.trunk = keras.Model(inp, {h: self.m.get_layer(h)(x) for h in self.heads})

    def projection(self):
        """-> (W (n_feat, P), b (P)) applied to RAW features: normalisation folded in."""
        K = self.m.get_layer("proj").kernel.numpy().astype(np.float64)
        b = self.m.get_layer("proj").bias.numpy().astype(np.float64)
        W = K / self.scale[:, None]
        return W, b - (self.mean / self.scale) @ K

    def windows(self, s, order=None, batch=4096):
        """Trunk inputs (pooled projections) at the split's valid centres."""
        ds = mres.MResWindows(s, self.mean.astype(np.float32), self.scale.astype(np.float32),
                              self.spec, targets=(), batch=batch)
        for i in range(len(ds)):
            yield ds.order[i * batch:(i + 1) * batch], self.pre.predict_on_batch(ds[i][0])


def representative(stage, splits, batches=2, seed=0):
    """Pooled projections from a shuffled sample of every calibration corpus."""
    rows = []
    for i, s in enumerate(splits):
        ds = mres.MResWindows(s, stage.mean.astype(np.float32), stage.scale.astype(np.float32),
                              stage.spec, targets=(), batch=512, shuffle=True, seed=seed + i)
        for j in range(min(batches, len(ds))):
            rows.append(stage.pre.predict_on_batch(ds[j][0]))
    rows = np.concatenate(rows).astype(np.float32)
    np.random.default_rng(seed).shuffle(rows)

    def gen():
        for r in rows:
            yield [r[None, :]]
    return gen


class Int8:
    def __init__(self, blob, heads):
        self.it = tf.lite.Interpreter(model_content=blob)
        self.inp = self.it.get_input_details()[0]
        self.outs = sorted(self.it.get_output_details(),
                           key=lambda d: int(d["name"].rsplit(":", 1)[1]))
        self.heads = heads
        self.q_in = self.inp["quantization"]
        self.batch = None

    def quantise(self, x):
        s, z = self.q_in
        return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)

    def __call__(self, x):
        if self.batch != len(x):
            self.it.resize_tensor_input(self.inp["index"], [len(x), x.shape[1]])
            self.it.allocate_tensors()
            self.batch = len(x)
        self.it.set_tensor(self.inp["index"], self.quantise(x))
        self.it.invoke()
        out = {}
        for h, d in zip(self.heads, self.outs):
            s, z = d["quantization"]
            q = self.it.get_tensor(d["index"])
            out[h] = (q.astype(np.float32) - z) * s, q
        return out


def pipeline(A, B, s_raw, tA=None, tB=None):
    """Beat (and other heads) of B over a mel=raw split, float or int8 per stage."""
    n = len(s_raw.X)
    sA = replace(s_raw, X=data.mel_view(s_raw.X, s_raw.take, "flux16"))
    act = np.zeros(n, np.float32)
    for c, x in A.windows(sA):
        act[c] = (tA(x)["beat"][0][:, 0] if tA else A.trunk.predict_on_batch(x)["beat"][:, 0])
    fb = closedloop.fb_series(act, s_raw.take)
    sB = replace(sA, X=np.concatenate([sA.X, fb[:, None]], axis=1))
    out = {h: np.zeros(n, np.float32) for h in ("beat", "beat_offset", "music")}
    for c, x in B.windows(sB):
        y = tB(x) if tB else {h: (v, None) for h, v in B.trunk.predict_on_batch(x).items()}
        for h in out:
            if h in y:
                out[h][c] = (y[h][0] if tB else y[h][0])[:, 0]
    return act, out


def score_sets(A, B, tA, tB, eval_start=461, sets=None):
    res = {}
    for name, path in TEST.items():
        if sets and name not in sets:
            continue
        s = data.load_many([Path(path)], "test", mel="raw")
        row = {}
        for lbl, (a8, b8) in (("float32", (None, None)), ("int8", (tA, tB))):
            _, out = pipeline(A, B, s, a8, b8)
            act = np.where(compare.pos_in_take(s) >= eval_start, out["beat"], 0.0)
            row[lbl] = (gridmetrics.per_take(act, s) if name in ("clips", "loops_grid", "melodic_grid")
                        else compare.tempo_score(act, s))
            row[lbl + "_act"] = act
        c = s.centres
        row["correlation"] = float(np.corrcoef(row["float32_act"][c], row["int8_act"][c])[0, 1])
        del row["float32_act"], row["int8_act"]
        res[name] = row
        print(name, json.dumps(row), flush=True)
    return res


def mel16_filterbank():
    """The 40-band triangles merged into the 16 bands flux16 uses (np.array_split
    groups, power mean): mean(10**log10(P @ W_b)) over a group is P @ mean(W_b)."""
    W40 = features._mel_matrix().astype(np.float64)
    groups = np.array_split(np.arange(W40.shape[1]), 16)
    W16 = np.stack([W40[:, g].mean(axis=1) for g in groups], axis=1)
    rows = []
    for b in range(16):
        nz = np.flatnonzero(W16[:, b] > 0)
        rows.append({"start_bin": int(nz[0]), "weights": [float(v) for v in W16[nz[0]:nz[-1] + 1, b]]})
    return W16, rows


def device_pcm(song_id, start_s, seconds):
    from dataset import augment, render
    from dataset.songgrid import SONGS, decode
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    m = decode(SONGS / index[song_id]["audio"])
    m = m[int(start_s * SAMPLE_RATE): int((start_s + seconds) * SAMPLE_RATE)]
    rng = np.random.default_rng(1)
    y = augment.finish(augment.apply_ir(m, rng), 240, rng, dbfs=float(np.mean(render.LEVEL_DBFS)))
    return (y * 32767.0).astype(np.int16)


def golden(A, B, tA, tB, pcm, W16):
    """Device-order walk from boot over `pcm`: every block's features, mel flux,
    projections, A's beat, fb3 and B's outputs."""
    X = features.featurise(pcm, hicut=True)[:, :12]
    lm = features.log_mel(pcm)
    n = len(X)
    # the firmware formula: 16-band filterbank on the power spectrum
    x = np.concatenate([np.zeros(features.MEL_WIN - BLOCK_SAMPLES, np.float32),
                        pcm[: n * BLOCK_SAMPLES].astype(np.float32) / 32768.0])
    frames = np.lib.stride_tricks.sliding_window_view(x, features.MEL_WIN)[::BLOCK_SAMPLES][:n]
    P = np.abs(np.fft.rfft(frames * np.hanning(features.MEL_WIN), axis=1)) ** 2
    lb16 = np.log10(P @ W16 + 1e-10)
    flux = np.zeros_like(lb16)
    flux[1:] = np.maximum(lb16[1:] - lb16[:-1], 0.0)
    take = np.zeros(n, np.int32)
    raw = np.concatenate([X, lm], axis=1)
    view = data.mel_view(raw, take, "flux16")
    assert np.abs(view[:, 12:] - flux).max() < 1e-4, np.abs(view[:, 12:] - flux).max()
    s = data.Split(raw, np.zeros((n, 4), np.float32), np.zeros((n, 3), np.float32), take,
                   np.arange(n))
    act_q, outB = pipeline(A, B, s, tA, tB)
    fb = closedloop.fb_series(act_q, take)
    WA, bA = A.projection()
    WB, bB = B.projection()
    projA = view @ WA + bA
    projB = np.concatenate([view, fb[:, None]], axis=1) @ WB + bB
    cen = mres.valid_centres(take, A.spec)
    # full int8 input vectors for one centre, as the single-stage golden had
    c0 = int(cen[len(cen) // 2])
    sA = replace(s, X=view)
    sB = replace(s, X=np.concatenate([view, fb[:, None]], axis=1))
    xa = next(x[list(c).index(c0)] for c, x in A.windows(sA) if c0 in c)
    xb = next(x[list(c).index(c0)] for c, x in B.windows(sB) if c0 in c)
    ya, yb = tA(xa[None]), tB(xb[None])
    return {
        "blocks": n, "first_centre": int(cen[0]),
        "fe": np.round(X, 6).tolist(), "mel_flux16": np.round(flux, 6).tolist(),
        "proj_a": np.round(projA, 5).tolist(), "a_beat_int8_dequant": np.round(act_q, 6).tolist(),
        "fb3": np.round(fb, 6).tolist(), "proj_b": np.round(projB, 5).tolist(),
        "b_beat": np.round(outB["beat"], 6).tolist(),
        "b_beat_offset": np.round(outB["beat_offset"], 6).tolist(),
        "b_music": np.round(outB["music"], 6).tolist(),
        "check_centre": c0,
        "a_input_int8": tA.quantise(xa[None])[0].tolist(),
        "a_output_int8": {h: v[1][0].tolist() for h, v in ya.items()},
        "b_input_int8": tB.quantise(xb[None])[0].tolist(),
        "b_output_int8": {h: v[1][0].tolist() for h, v in yb.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--golden-song", default=None, help="song id; default: first confirmed label")
    ap.add_argument("--golden-start", type=float, default=60.0)
    ap.add_argument("--golden-seconds", type=float, default=12.0)
    ap.add_argument("--calib", nargs="+", default=list(CALIB))
    ap.add_argument("--sets", nargs="*", default=None, help="subset of the test sets, for a smoke test")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    A = Stage(a.a, heads=["beat"])
    B = Stage(a.b)
    assert B.info.get("cascade") == [f"fb3_{a.a.name}"], B.info.get("cascade")

    print("calibrating...", flush=True)
    calA = [data.load(Path(c), "train", mel="flux16") for c in a.calib]
    blobA = to_tflite(A.trunk, representative(A, calA))
    calB = [data.load(Path(c), "train", mel="flux16", aux=[f"fb3_{a.a.name}"]) for c in a.calib]
    blobB = to_tflite(B.trunk, representative(B, calB))
    del calA, calB
    (a.out / "stage_a.tflite").write_bytes(blobA)
    (a.out / "stage_b.tflite").write_bytes(blobB)
    tA, tB = Int8(blobA, A.heads), Int8(blobB, B.heads)
    print(f"stage A {len(blobA):,} B, stage B {len(blobB):,} B", flush=True)

    metrics = score_sets(A, B, tA, tB, sets=a.sets)
    W16, fb_rows = mel16_filterbank()
    WA, bA = A.projection()
    WB, bB = B.projection()

    songs = sorted(p.stem for p in Path("songs/labels").glob("*.json")
                   if json.loads(p.read_text())["status"] == "confirmed")
    gid = a.golden_song or songs[0]
    pcm = device_pcm(gid, a.golden_start, a.golden_seconds)
    (a.out / "golden").mkdir(exist_ok=True)
    (a.out / "golden" / "pcm_s16le_48k.raw").write_bytes(pcm.astype("<i2").tobytes())
    g = golden(A, B, tA, tB, pcm, W16)
    g.update(song=gid, start_s=a.golden_start, seconds=a.golden_seconds,
             pcm="pcm_s16le_48k.raw: mono int16 little endian, 48 kHz, fed to the device chain from boot")
    (a.out / "golden" / "golden.json").write_text(json.dumps(g))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta_fe = data.meta(Path("data/features4_fe3"))

    def q(t):
        s, z = t.q_in
        outs = {h: {"tensor": d["name"], "scale": float(d["quantization"][0]),
                    "zero_point": int(d["quantization"][1]), "shape": [int(v) for v in d["shape"][1:]]}
                for h, d in zip(t.heads, t.outs)}
        return {"input_scale": float(s), "input_zero_point": int(z), "outputs": outs}

    spec = A.spec
    meta = {
        "format": "two-stage-1",
        "fe_spec_version": meta_fe["fe_spec_version"], "fe_variant": meta_fe.get("fe_variant"),
        "fe_variant_name": meta_fe.get("fe_variant_name"),
        "block_samples": BLOCK_SAMPLES, "sample_rate": SAMPLE_RATE,
        "features": {
            "order": list(features.FEATURE_ORDER) + [f"mel_flux{i}" for i in range(16)],
            "fe": "fe_out_t fields as before, 12",
            "mel": {"input": "the same int16 block fed to fe_run, / 32768",
                    "window": features.MEL_WIN, "window_fn": "Hann, symmetric (numpy.hanning)",
                    "window_end": "the last sample of the current block (previous block + current)",
                    "fft": features.MEL_WIN, "power": "|rfft|^2, 513 bins",
                    "bands": 16, "filterbank": fb_rows,
                    "log": "log10(power @ filter + 1e-10)",
                    "flux": "max(0, log_t - log_(t-1)) per band; 0 on the first block after boot",
                    "source_triangles": {"bands": features.MEL_BANDS, "lo_hz": features.MEL_LO_HZ,
                                         "hi_hz": features.MEL_HI_HZ, "mel": "HTK",
                                         "merged": "numpy.array_split(40, 16), mean of filters"}}},
        "context": {**context_spec(A.info["spec"]), "frames": spec.frames,
                    "inputs_per_stage": spec.width(12), "lookback_blocks": spec.lookback,
                    "ring_blocks": spec.lookback + 1 + spec.fine_future,
                    "lookahead_blocks": spec.fine_future},
        "stage_a": {"file": "stage_a.tflite", "inputs": 28,
                    "projection": {"weights": np.round(WA, 8).tolist(), "bias": np.round(bA, 8).tolist(),
                                   "applied_to": "raw features (28), normalisation folded in"},
                    "outputs": A.heads, **q(tA), "macs_trunk": int(sum(np.prod(l.kernel.shape)
                                                                     for l in A.trunk.layers if hasattr(l, "kernel"))),
                    "tflite_bytes": len(blobA), "sha256": hashlib.sha256(blobA).hexdigest()},
        "stage_b": {"file": "stage_b.tflite", "inputs": 29,
                    "feedback": {"input_index": 28, "value": "stage A beat output, dequantised, 0..1",
                                 "shift_blocks": closedloop.SHIFT,
                                 "rule": "feature 28 of block j = stage A beat for centre j-3; 0 until A has produced it"},
                    "projection": {"weights": np.round(WB, 8).tolist(), "bias": np.round(bB, 8).tolist(),
                                   "applied_to": "raw features (28) + feedback, normalisation folded in"},
                    "outputs": B.heads, "hit_classes": HIT_CLASSES, **q(tB),
                    "macs_trunk": int(sum(np.prod(l.kernel.shape) for l in B.trunk.layers if hasattr(l, "kernel"))),
                    "tflite_bytes": len(blobB), "sha256": hashlib.sha256(blobB).hexdigest()},
        "macs_per_block": {"projections": 28 * 12 + 29 * 12,
                           "stage_a": None, "stage_b": None},
        "metrics": metrics,
        "trained_from": {"repo_commit": commit, "manifest_sha256": meta_fe.get("manifest_sha256"),
                         "ir_sha256": meta_fe.get("ir_sha256"), "stage_a_run": str(a.a), "stage_b_run": str(a.b)},
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta["macs_per_block"]["stage_a"] = meta["stage_a"]["macs_trunk"]
    meta["macs_per_block"]["stage_b"] = meta["stage_b"]["macs_trunk"]
    meta["macs_per_block"]["total"] = sum(v for v in meta["macs_per_block"].values() if v)
    (a.out / "model_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["macs_per_block"]), "golden", gid)


if __name__ == "__main__":
    main()
