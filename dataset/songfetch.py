"""Fetch the songs in songs/likes.csv (a Spotify export) from YouTube.

Each track is searched as "artist - title" and the candidate is chosen by
duration against Spotify's, since the grid is labelled on the downloaded file
and a different edit or a live take would not be the listed track anyway.
Auto-generated "Artist - Topic" uploads are the album audio and win ties.

    python -m dataset.songfetch            # resumable; skips what is in the index
    python -m dataset.songfetch --retry    # retry not_found / error rows

-> songs/audio/<spotify id>.opus, one row per track in songs/index.jsonl.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import yt_dlp

SONGS = Path("songs")
SEARCH_N = 8
DUR_TOL_S = 4.0
DUR_TOL_FRAC = 0.02
# Versions that are not the listed recording, unless the listed title says so.
OTHER_VERSION = ("live", "remix", "cover", "slowed", "sped up", "reverb", "8d", "nightcore",
                 "extended", "edit", "instrumental", "karaoke", "acoustic", "mix)", "hour")


def tracks(csv_path: Path) -> list[dict]:
    out = []
    with open(csv_path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "id": r["Track URI"].rsplit(":", 1)[-1],
                "title": r["Track Name"],
                "artist": r["Artist Name(s)"],
                "album": r["Album Name"],
                "duration_s": int(r["Duration (ms)"]) / 1000.0,
                "spotify_tempo": float(r["Tempo"] or 0),
                "time_signature": int(r["Time Signature"] or 0),
                "genres": r["Genres"],
            })
    return out


def score(entry: dict, t: dict) -> float | None:
    """Lower is better; None rejects the candidate."""
    d = entry.get("duration")
    if not d:
        return None
    diff = abs(d - t["duration_s"])
    if diff > max(DUR_TOL_S, DUR_TOL_FRAC * t["duration_s"]):
        return None
    title = (entry.get("title") or "").lower()
    listed = t["title"].lower()
    if any(w in title and w not in listed for w in OTHER_VERSION):
        return None
    words = [w for w in re.findall(r"\w+", listed) if len(w) > 2]
    missing = sum(w not in title for w in words) / max(len(words), 1)
    topic = (entry.get("channel") or "").endswith(" - Topic")
    return diff + 10.0 * missing - (3.0 if topic else 0.0)


def fetch(t: dict, audio_dir: Path) -> dict:
    row = dict(t, status="not_found")
    artist = re.split(r"[;,]", t["artist"])[0].strip()
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True,
            "js_runtimes": {"node": {}}}
    with yt_dlp.YoutubeDL(opts) as y:
        res = y.extract_info(f"ytsearch{SEARCH_N}:{artist} - {t['title']}", download=False)
    cands = [(s, e) for e in res.get("entries") or () if (s := score(e, t)) is not None]
    if not cands:
        return row
    _, e = min(cands, key=lambda c: c[0])
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "format": "bestaudio/best",
            "outtmpl": str(audio_dir / f"{t['id']}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "opus"}],
            "js_runtimes": {"node": {}}}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([f"https://www.youtube.com/watch?v={e['id']}"])
    row.update(status="ok", audio=f"audio/{t['id']}.opus", youtube_id=e["id"],
               youtube_title=e.get("title"), youtube_channel=e.get("channel"),
               youtube_duration_s=e.get("duration"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=SONGS / "likes.csv")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    audio = SONGS / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    idx_path = SONGS / "index.jsonl"
    index = {}
    if idx_path.exists():
        for l in idx_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                index[r["id"]] = r
    todo = [t for t in tracks(a.csv) if t["id"] not in index
            or (a.retry and index[t["id"]]["status"] != "ok")]
    if a.limit:
        todo = todo[: a.limit]
    for n, t in enumerate(todo):
        try:
            r = fetch(t, audio)
        except Exception as ex:  # one bad video must not stop the batch
            r = dict(t, status="error", error=str(ex)[:300])
        index[t["id"]] = r
        idx_path.write_text("".join(json.dumps(v) + "\n" for v in index.values()))
        print(f"{n+1}/{len(todo)} {r['status']:9} {t['artist'][:24]:24} {t['title'][:40]:40} "
              f"{r.get('youtube_channel') or ''}", flush=True)
        time.sleep(1.0)
    ok = sum(r["status"] == "ok" for r in index.values())
    print(f"{ok}/{len(index)} ok")


if __name__ == "__main__":
    main()
