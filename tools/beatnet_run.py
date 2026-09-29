"""Run BeatNet (Heydari et al., ISMIR 2021; github.com/mjhydri/BeatNet, CC-BY-4.0)
over songs, for comparison and as a candidate teacher.

Runs in its own venv (~/.venvs/beatnet: torch CPU, madmom from git, BeatNet;
pyaudio is stubbed, only streaming mode needs it), not the project's.

    ~/.venvs/beatnet/bin/python tools/beatnet_run.py --ids-file ids.txt --cond clean
    ~/.venvs/beatnet/bin/python tools/beatnet_run.py --ids-file ids.txt --cond device \
        --audio-dir cache/beatnet_device

-> songs/beatnet/<cond>/<id>.npz: offline (DBN) and online (particle filter)
beats [time s, position in the bar, 1 = downbeat], and the network's beat/downbeat activation at 50 fps.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

SR = 22050
SONGS = Path("songs")


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--cond", default="clean", help="clean: the file; device: pre-rendered chain")
    ap.add_argument("--audio-dir", type=Path, default=None, help="<id>.npy at 22.05 kHz for --cond device")
    ap.add_argument("--model", type=int, default=1)
    ap.add_argument("--skip-online", action="store_true")
    a = ap.parse_args()
    from BeatNet.BeatNet import BeatNet
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    out_dir = SONGS / "beatnet" / a.cond
    out_dir.mkdir(parents=True, exist_ok=True)
    off = BeatNet(a.model, mode="offline", inference_model="DBN", plot=[], thread=False)
    for n, k in enumerate(a.ids_file.read_text().split()):
        dst = out_dir / f"{k}.npz"
        if dst.exists():
            continue
        t = time.time()
        x = (np.load(a.audio_dir / f"{k}.npy").astype(np.float32) if a.cond != "clean"
             else decode(SONGS / index[k]["audio"]))
        act = off.activation_extractor_online(x)
        beats_off = off.estimator(act)
        res = {"offline": np.asarray(beats_off, np.float64).reshape(-1, 2),
               "activation": act.astype(np.float32), "fps": 50}
        if not a.skip_online:
            # A fresh estimator per song: the particle filter keeps state between calls.
            on = BeatNet(a.model, mode="online", inference_model="PF", plot=[], thread=False)
            res["online"] = np.asarray(on.process(x), np.float64).reshape(-1, 2)
        np.savez(dst, **res)
        print(f"{n+1} {k} {len(x)/SR:.0f}s audio  {len(res['offline'])} beats  "
              f"{time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
