"""Per-block inputs derived from a frozen model A, for a model B that sees them.

Model B gets what A produced, including A's errors, so it learns how far to
trust it; true beats would make it copy the feedback and ignore the audio.

    act_<A>: A's beat activation, one column. Its newest context frames (t+1,
             t+2) need audio up to t+4: two blocks more lookahead than A had.
    fb3_<A>: the activation shifted by 3 blocks (train.closedloop), what a model
             fed its own past output can see; the self-feedback runs train on it.
    st_<A>:  the streaming stage 2 (train.stream, hold, 8 s) run on A's
             activation, causal: log2 period, sin/cos of the beat phase at this
             block, and the held grid's recent contrast. Zeros before the first fit.

Written as data/<corpus>/aux_<tag>_<split>.npy and appended by data.load(aux=...).

    python -m train.cascade runs/n_song_s0 data/features4_mel data/noise_mel ...
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np

from . import stream

LAG_REF = 45.0  # blocks, ~125 BPM: log2(lag / LAG_REF) sits near 0


def state(act: np.ndarray) -> np.ndarray:
    out = np.zeros((len(act), 4), np.float32)
    res = stream.track(act, hold=True)
    for i, (t, g) in enumerate(res):
        if g is None:
            continue
        # The grid fitted at update t applies from block t until the next update.
        b = np.arange(t, min(res[i + 1][0] if i + 1 < len(res) else len(act), len(act)))
        ph = 2 * np.pi * (((b - g.anchor) / g.lag) % 1.0)
        out[b, 0] = np.log2(g.lag / LAG_REF)
        out[b, 1], out[b, 2] = np.sin(ph), np.cos(ph)
        out[b, 3] = stream.contrast(act, g, max(0, t - stream.RECENT), t)
    return out


def _state_take(args):
    act, margin, floor, recent = args
    stream.MARGIN, stream.FLOOR, stream.RECENT = margin, floor, recent
    return state(act)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--what", nargs="+", default=["act", "fb3", "st"])
    ap.add_argument("--st-tag", default="st", help="prefix for a tracker variant")
    ap.add_argument("--margin", type=float, default=stream.MARGIN)
    ap.add_argument("--floor", type=float, default=stream.FLOOR)
    ap.add_argument("--recent", type=int, default=stream.RECENT)
    a = ap.parse_args()
    from . import gpu  # noqa: F401  must precede keras
    import json

    import keras

    from . import data, scoreboard, seq
    r = Path(a.run)
    info = json.loads((r / "result.json").read_text())
    m = keras.models.load_model(r / "model.keras", compile=False)
    nz = np.load(r / "norm.npz")
    tag = r.name
    from . import closedloop
    for c in a.corpora:
        for sp in a.splits:
            if not (Path(c) / f"{sp}.npz").exists():
                continue
            s = data.load(Path(c), sp, mel=info.get("mel"))
            act = scoreboard.activation(m, s, nz["mean"], nz["scale"], scoreboard.run_spec(r))[0]
            if "act" in a.what:
                np.save(Path(c) / f"aux_act_{tag}_{sp}.npy", act[:, None].astype(np.float32))
            if "fb3" in a.what:
                np.save(Path(c) / f"aux_fb3_{tag}_{sp}.npy",
                        closedloop.fb_series(act, s.take)[:, None].astype(np.float32))
            if "st" in a.what:
                groups = seq.takes(s)
                args = [(act[g], a.margin, a.floor, a.recent) for g in groups]
                # spawn, not fork: the parent holds a CUDA context
                with multiprocessing.get_context("spawn").Pool(a.procs) as pool:
                    parts = pool.map(_state_take, args, chunksize=8)
                np.save(Path(c) / f"aux_{a.st_tag}_{tag}_{sp}.npy", np.concatenate(parts))
            print(f"{c} {sp}: {len(act):,} blocks", flush=True)


if __name__ == "__main__":
    main()
