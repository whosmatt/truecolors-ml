"""Playlists of hand-labelled songs: tempo changes the way they happen in use.

Most tempo changes a light sees are track changes. Each take chains excerpts of
labelled songs with a hard cut, a cut with a silence gap, or an equal-power
crossfade, so a tracker is scored on letting go of a grid as well as holding it.
Songs stay in their songset split (by id hash), so test playlists only contain
test songs.

Labels come from the confirmed grids. Masked (-1): the crossfade overlap, spans
excluded in the labeller, and songs rejected there. Silence gaps are labelled
"no beat" and not music.

    python -m dataset.playlist --out data/playlists
    TC_MEL=1 python -m dataset.playlist --out data/playlists_mel

Writes <split>.npz like the other corpora, plus playlists.json: every take's
segments (song, block span, bpm) and transitions (kind, block).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frontend.fe import BLOCK_SAMPLES, SAMPLE_RATE

from . import augment, features, gridset, render
from .songgrid import SONGS, decode
from .songset import MASK, labels, split_of

SR = SAMPLE_RATE
PER_PLAYLIST = 6
EXCERPT_S = (40.0, 90.0)
KINDS = (("cut", 0.4), ("gap", 0.2), ("fade", 0.4))
GAP_S = (0.5, 2.5)
FADE_S = (3.0, 12.0)
GAIN_DB = (-4.0, 2.0)
COUNTS = {"train": 80, "val": 12, "test": 20}


def song_beats(lab: dict, dur_s: float) -> tuple[np.ndarray, list]:
    """-> (beat times s on the song's timeline, excluded spans); no beats if rejected."""
    if lab["status"] != "confirmed":
        return None, []
    T = 60.0 / lab["bpm"]
    k = np.arange(np.ceil(-lab["t0_s"] / T), np.floor((dur_s - lab["t0_s"]) / T) + 1)
    ex = [(s["start_s"], s["end_s"]) for s in lab.get("sections") or () if not s.get("include", True)]
    return lab["t0_s"] + k * T, ex


def build_one(songs: list, rng: np.random.Generator, index: dict):
    """-> (mono audio, beat sample positions, masked sample spans, gap spans, info)."""
    order = rng.choice(len(songs), size=min(PER_PLAYLIST, len(songs)), replace=False)
    parts, beats, masked, gaps, segs, trans = [], [], [], [], [], []
    pos = 0  # start of the next excerpt, samples
    for j, i in enumerate(order):
        lab = songs[i]
        audio = decode(SONGS / index[lab["id"]]["audio"])  # decoded per use: all songs are ~4 GB
        dur = len(audio) / SR
        L = min(float(rng.uniform(*EXCERPT_S)), dur)
        s0 = float(rng.uniform(0.0, dur - L))
        x = audio[int(s0 * SR): int((s0 + L) * SR)] * 10 ** (rng.uniform(*GAIN_DB) / 20)
        kind = None
        if j:
            kind = KINDS[rng.choice(len(KINDS), p=[p for _, p in KINDS])][0]
            if kind == "gap":
                g = int(rng.uniform(*GAP_S) * SR)
                gaps.append((pos, pos + g))
                pos += g
            elif kind == "fade":
                f = min(int(rng.uniform(*FADE_S) * SR), len(x) // 3, len(parts[-1][1]) // 3)
                pos -= f
                ramp = np.linspace(0.0, np.pi / 2, f, dtype=np.float32)
                x = x.copy()
                x[:f] *= np.sin(ramp)
                prev = parts[-1][1]
                prev[len(prev) - f:] *= np.cos(ramp)
                masked.append((pos, pos + f))
            trans.append({"kind": kind, "sample": pos})
        bt, ex = song_beats(lab, dur)
        if bt is None:
            masked.append((pos, pos + len(x)))
        else:
            sel = bt[(bt >= s0) & (bt < s0 + L)]
            beats.append(pos + ((sel - s0) * SR).astype(np.int64))
            for a, b in ex:
                a, b = max(a, s0), min(b, s0 + L)
                if b > a:
                    masked.append((pos + int((a - s0) * SR), pos + int((b - s0) * SR)))
        segs.append({"song": lab["id"], "status": lab["status"], "bpm": lab.get("bpm"),
                     "start_sample": pos, "end_sample": pos + len(x), "from_s": s0})
        parts.append((pos, x))
        pos += len(x)
    y = np.zeros(pos, np.float32)
    for p, x in parts:
        y[p: p + len(x)] += x
    return y, np.concatenate(beats) if beats else np.zeros(0, np.int64), masked, gaps, {
        "segments": segs, "transitions": trans}


def block_labels(n, beats, masked, gaps, segs):
    beat = np.zeros((n, 1), np.float32)
    frac = np.zeros((n, 1), np.float32)
    b = beats / BLOCK_SAMPLES
    b = b[b < n]
    beat[b.astype(int), 0] = 1.0
    frac[b.astype(int), 0] = b % 1.0
    period = np.full((n, 1), MASK, np.float32)
    for s in segs:
        if s["bpm"]:
            period[s["start_sample"] // BLOCK_SAMPLES: s["end_sample"] // BLOCK_SAMPLES] = \
                60.0 / s["bpm"] * SR / BLOCK_SAMPLES
    music = np.ones((n, 1), np.float32)
    for a, e in masked:
        sl = slice(a // BLOCK_SAMPLES, -(-e // BLOCK_SAMPLES))
        beat[sl], frac[sl], period[sl] = MASK, MASK, MASK
    for a, e in gaps:
        sl = slice(-(-a // BLOCK_SAMPLES), e // BLOCK_SAMPLES)
        music[sl], period[sl] = 0.0, MASK
    return beat, frac, period, music


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/playlists"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=None, help="as dataset.songset --fold")
    ap.add_argument("--train-count", type=int, default=COUNTS["train"])
    a = ap.parse_args()
    splitter = split_of
    if a.fold is not None:
        from . import songfolds
        folds = songfolds.load()
        splitter = lambda k: songfolds.split_for(k, a.fold, folds)
    counts = dict(COUNTS, train=a.train_count)
    a.out.mkdir(parents=True, exist_ok=True)
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    pool = labels(auto=False)
    skip = int(gridset.SETTLE_S * SR / BLOCK_SAMPLES)
    keys = ("X", "y", "off", "beat", "beat_off", "period", "music", "take", "notch")
    info, n, t_start = {}, 0, time.time()
    for split, count in counts.items():
        songs = [lab for lab in pool if splitter(lab["id"]) == split]
        if not songs:  # --fold -1: every song trains, none tests
            continue
        rng = np.random.default_rng(a.seed * 1000 + len(split))
        acc = {k: [] for k in keys}
        info[split] = []
        for p in range(count):
            y, beats, masked, gaps, meta = build_one(songs, rng, index)
            notch = features.NOTCH_HZ[p % len(features.NOTCH_HZ)]
            y = augment.finish(augment.apply_ir(y), notch, rng, dbfs=float(rng.uniform(*render.LEVEL_DBFS)))
            X = features.featurise((y * 32767.0).astype(np.int16), hicut=True)
            L = len(X)
            beat, frac, period, music = block_labels(L, beats, masked, gaps, meta["segments"])
            for k, v in (("X", X), ("beat", beat), ("beat_off", frac), ("period", period),
                         ("music", music)):
                acc[k].append(v[skip:])
            acc["y"].append(np.full((L - skip, 4), MASK, np.float32))
            acc["off"].append(np.full((L - skip, 3), MASK, np.float32))
            acc["take"].append(np.full(L - skip, n, np.int32))
            acc["notch"].append(np.full(L - skip, notch, np.int16))
            to_block = lambda smp: int(smp // BLOCK_SAMPLES) - skip
            info[split].append({
                "take": n,
                "segments": [dict(s, start_block=to_block(s["start_sample"]),
                                  end_block=to_block(s["end_sample"])) for s in meta["segments"]],
                "transitions": [dict(t, block=to_block(t["sample"])) for t in meta["transitions"]]})
            n += 1
            print(f"  {split} {p+1}/{count}  {L/93.75/60:.1f} min  {time.time()-t_start:.0f}s",
                  flush=True)
        arrs = {k: np.concatenate(v) for k, v in acc.items()}
        np.savez(a.out / f"{split}.npz", **arrs)
        print(f"{split:6} {len(arrs['X']):9,} blocks  {count} playlists from {len(songs)} songs")
    (a.out / "playlists.json").write_text(json.dumps(info))
    (a.out / "meta.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "playlists of hand-labelled songs",
        "masked": ["y", "off"], "per_playlist": PER_PLAYLIST, "excerpt_s": EXCERPT_S,
        "kinds": dict(KINDS), "gap_s": GAP_S, "fade_s": FADE_S, "counts": counts, "fold": a.fold,
        **features.spec(hicut=True)}, indent=2))


if __name__ == "__main__":
    main()
