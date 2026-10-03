"""Songs through the device chain (mic IR, coil whine, level; the same seed and
setting as dataset.songgrid), resampled for an external model, e.g. BeatNet at 22.05 kHz.

    python tools/render_device.py ids.txt --out cache/beatnet_device --rate 22050
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import augment, render  # noqa: E402
from dataset.songgrid import SONGS, decode  # noqa: E402
from frontend.fe import SAMPLE_RATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=int, default=22050)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    for k in a.ids.read_text().split():
        dst = a.out / f"{k}.npy"
        if dst.exists():
            continue
        m = decode(SONGS / index[k]["audio"])
        rng = np.random.default_rng(int.from_bytes(k.encode()[:4], "big"))
        y = augment.finish(augment.apply_ir(m, rng), 240, rng, dbfs=float(np.mean(render.LEVEL_DBFS)))
        np.save(dst, soxr.resample(y, SAMPLE_RATE, a.rate, quality="HQ").astype(np.float16))
        print(k, flush=True)


if __name__ == "__main__":
    main()
