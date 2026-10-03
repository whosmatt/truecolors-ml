"""Music gate of a two-stage export: per-take median of stage B's int8 music
output over val takes, share kept (music) or rejected (non-music, silence) at a
threshold, in results/8's groups plus songs.

    python -m train.gate2 export/twostage1 --out export/twostage1/gate.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import data, export2, grid5, seq

GROUPS = [("real loops", "data/loops_grid_fe3", 1), ("melodic", "data/melodic_fe3", 1),
          ("rendered", "data/features4_fe3", 1), ("songs", "data/songs_all_fe3", 1),
          ("non-music", "data/noise_fe3", 0), ("silence", "data/noise_fe3", -1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.8])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    meta = json.loads((a.export / "model_meta.json").read_text())
    A = export2.Stage(Path(meta["trained_from"]["stage_a_run"]), heads=["beat"])
    B = export2.Stage(Path(meta["trained_from"]["stage_b_run"]))
    tA = export2.Int8((a.export / "stage_a.tflite").read_bytes(), A.heads)
    tB = export2.Int8((a.export / "stage_b.tflite").read_bytes(), B.heads)
    cache, res = {}, {}
    for name, path, want in GROUPS:
        if path not in cache:
            s = data.load_many([Path(path)], "val", mel="raw")
            _, out = export2.pipeline(A, B, s, tA, tB)
            sil = grid5.silent(s)
            rows = []
            for g in seq.takes(s):
                c = g[g >= g[0] + 271]  # blocks with a full ring
                if len(c):
                    rows.append((float(np.median(out["music"][c])), float(np.mean(sil[c]))))
            cache[path] = np.array(rows)
        r = cache[path]
        if want == 0:
            r = r[r[:, 1] < 0.5]
        elif want == -1:
            r = r[r[:, 1] >= 0.5]
        res[name] = {"takes": int(len(r)), **{
            f"{t}": float(np.mean(r[:, 0] >= t) if want == 1 else np.mean(r[:, 0] < t)) for t in a.thresholds}}
        print(name, json.dumps(res[name]), flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
