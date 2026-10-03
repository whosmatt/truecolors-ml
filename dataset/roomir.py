"""Precompute the augmentation kernels: every room IR channel convolved with the
device mic's own response, so a take costs one convolution instead of two.

    python -m dataset.roomir            # -> data/ir/kernels.npz

Rooms come from data/ir/rooms/manifest.json (originals, mixed rates and
channels); the mic from data/ir/mic_only.wav (minimum phase, peak at 0). Each
room is trimmed to its direct sound so the kernel adds no onset delay, and cut
where its decay meets its own noise floor, so recording noise is not convolved
into every take.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 48000
ROOMS = Path("data/ir/rooms")
MIC = Path("data/ir/mic_only.wav")
OUT = Path("data/ir/kernels.npz")
ONSET = 0.3          # direct sound: first sample at this fraction of the peak
FLOOR_MARGIN_DB = 3.0
MAX_S = 1.5          # longest kept tail; selection is RT <= 1.2 s
FADE_S = 0.01
# Labels assume the first arrival dominates. Drop kernels whose peak comes >10 ms
# after a direct sound >6 dB below it: pre-delayed reverb programs, not rooms.
LATE_PEAK_S, WEAK_DIRECT_DB = 0.010, -6.0


def trim(h: np.ndarray) -> np.ndarray:
    a = np.abs(h)
    h = h[int(np.argmax(a >= ONSET * a.max())) :]
    w = int(0.01 * SR)
    env = 10 * np.log10(np.convolve(h**2, np.ones(w) / w, "same") + 1e-30)
    env -= env.max()
    floor = np.median(env[-max(len(env) // 10, w) :])
    cut_db = max(floor + FLOOR_MARGIN_DB, -70.0)
    i0 = int(0.02 * SR)
    below = np.where(env[i0:] < cut_db)[0]
    end = min(i0 + int(below[0]) if len(below) else len(h), int(MAX_S * SR))
    h = h[:end].copy()
    f = min(int(FADE_S * SR), len(h) // 4)
    if f:
        h[-f:] *= np.linspace(1.0, 0.0, f)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    man = json.loads((ROOMS / "manifest.json").read_text())
    mic, sr = sf.read(MIC, dtype="float64")
    assert sr == SR
    mic = mic[int(np.argmax(np.abs(mic))) :]
    ks, names, dropped = [], [], []
    for e in man["irs"]:
        x, sr = sf.read(ROOMS / e["path"], dtype="float64", always_2d=True)
        if sr != SR:
            x = soxr.resample(x, sr, SR, quality="VHQ")
        chans = [0] if x.shape[1] == 1 or np.allclose(x[:, 0], x[:, -1]) else range(x.shape[1])
        for c in chans:
            k = np.convolve(trim(x[:, c]), mic)
            mag = np.abs(k)
            if np.argmax(mag) > LATE_PEAK_S * SR and 20 * np.log10(mag[: SR // 1000].max() / mag.max()) < WEAK_DIRECT_DB:
                dropped.append(f"{e['path']}#{c}")
                continue
            ks.append((k / mag.max()).astype(np.float32))
            names.append(f"{e['path']}#{c}")
    lens = np.array([len(k) for k in ks])
    np.savez(a.out, data=np.concatenate(ks), offsets=np.concatenate([[0], np.cumsum(lens)]),
             names=np.array(names), mic=str(MIC), rooms=str(ROOMS / "manifest.json"))
    print("dropped (late peak, weak direct):", dropped)
    print(f"{a.out}: {len(ks)} kernels, length ms min {lens.min() / 48:.0f} "
          f"median {np.median(lens) / 48:.0f} max {lens.max() / 48:.0f}")


if __name__ == "__main__":
    main()
