"""Exported two-stage models on real device recordings of a hand-labelled song:
int8 pipeline + streaming stage 2, scored like the songs (usable 0.5 s updates).

Each export is fed the front end its model_meta.json names (v2 compiled from the
submodule's history, the current version through frontend.fe), so an older model is judged on the
features it was deployed with.

    python -m train.realcheck export/twostage1 export/twostage2 \
        --rec data/songcheck/2026-10-03 --song 3urwWcJ0qooKo5UiqfeIae
"""

import argparse
import ctypes
import json
import os
from pathlib import Path

os.environ["TC_MEL"] = "0"  # log-mel is appended here explicitly

import numpy as np

from dataset import features
from dataset.songset import MASK, grid_labels
from frontend import fe

from . import data, export2, stream

BS, SR = fe.BLOCK_SAMPLES, fe.SAMPLE_RATE
V2_COMMIT = "74cba7b"


def frontend_v2():
    """fe_out_t under FE_SPEC_VERSION 2: the front end as of firmware 74cba7b
    (v4's deployment), compiled from the submodule's history."""
    import subprocess
    dst = Path(fe.__file__).parent / "build" / "src_v2"
    for rel in ("components/audio/frontend.c", "components/audio/include/frontend.h"):
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(subprocess.run(["git", "-C", str(fe._ROOT), "show", f"{V2_COMMIT}:{rel}"],
                                       capture_output=True, check=True).stdout)
    lib = ctypes.CDLL(str(fe._build(dst)))
    lib.fe_state_size.restype = ctypes.c_size_t
    lib.fe_spec_version.restype = ctypes.c_int
    lib.fe_init.argtypes = [ctypes.c_void_p]
    lib.fe_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    assert lib.fe_spec_version() == 2

    def run(x):
        st = ctypes.create_string_buffer(lib.fe_state_size())
        lib.fe_init(st)
        n = len(x) // BS
        out = np.zeros(n, fe.DTYPE)
        lib.fe_run(st, np.ascontiguousarray(x, "<i2").ctypes.data, n, out.ctypes.data)
        return out
    return run


def featurise(x: np.ndarray, version: int) -> np.ndarray:
    raw = frontend_v2()(x) if version == 2 else fe.Frontend().run(x)
    assert version in (2, fe.SPEC_VERSION)
    return np.concatenate([features.flatten(raw), features.log_mel(x)], axis=1).astype(np.float32)


def split_for(recs, version, lab):
    Xs, beats, fracs, periods, takes = [], [], [], [], []
    period = 60.0 / lab["bpm"] * SR / BS
    ex = [(s["start_s"], s["end_s"]) for s in lab.get("sections") or () if not s.get("include", True)]
    for i, (x, off_s) in enumerate(recs):
        X = featurise(x, version)
        L = len(X)
        # recording time t is song time t + off_s
        b, f = grid_labels(L, lab["bpm"], lab["t0_s"] - off_s, [(s - off_s, e - off_s) for s, e in ex])
        Xs.append(X), beats.append(b), fracs.append(f)
        periods.append(np.full((L, 1), period, np.float32))
        takes.append(np.full(L, i, np.int32))
    X, take = np.concatenate(Xs), np.concatenate(takes)
    n = len(X)
    idx = np.arange(n)
    ok = (idx >= data.PAST) & (idx < n - data.FUTURE)
    ok[data.PAST : n - data.FUTURE] &= take[: n - data.PAST - data.FUTURE] == take[data.PAST + data.FUTURE :]
    return data.Split(X, np.full((n, 4), MASK, np.float32), np.full((n, 3), MASK, np.float32), take,
                      np.flatnonzero(ok).astype(np.int64), np.concatenate(beats), np.concatenate(fracs),
                      np.concatenate(periods), music=np.ones((n, 1), np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exports", type=Path, nargs="+")
    ap.add_argument("--rec", type=Path, default=Path("data/songcheck/2026-10-03"))
    ap.add_argument("--song", default="3urwWcJ0qooKo5UiqfeIae")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    lab = json.loads(Path(f"songs/labels/{a.song}.json").read_text())
    align = json.loads((a.rec / "align.json").read_text())
    recs = [(np.fromfile(a.rec / f"rec_{k}.raw", dtype="<i2"), v["song_offset_s"]) for k, v in sorted(align.items())]
    res = {}
    for ex in a.exports:
        meta = json.loads((ex / "model_meta.json").read_text())
        ver = int(meta["fe_spec_version"])
        A = export2.Stage(Path(meta["trained_from"]["stage_a_run"]), heads=["beat"])
        B = export2.Stage(Path(meta["trained_from"]["stage_b_run"]))
        tA = export2.Int8((ex / "stage_a.tflite").read_bytes(), A.heads)
        tB = export2.Int8((ex / "stage_b.tflite").read_bytes(), B.heads)
        s = split_for(recs, ver, lab)
        _, out = export2.pipeline(A, B, s, tA, tB)
        sc = stream.score(out["beat"], s)
        beat = s.beat[:, 0] > 0
        c = s.centres
        r = {"fe_spec_version": ver, **sc,
             "beat_act_on_beats": float(out["beat"][c][beat[c]].mean()),
             "beat_act_elsewhere": float(out["beat"][c][~beat[c]].mean()),
             "music_median": float(np.median(out["music"][c]))}
        res[str(ex)] = r
        print(ex, json.dumps(r), flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
