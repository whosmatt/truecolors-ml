"""Build the dataset manifest from the Live index.

One JSON object per line. The first line is the header: schema, the index
fingerprint, the parameters every measurement was made under, and counts. Hash
the file to identify a dataset; that hash goes in the model's model_meta.json.

    python -m dataset.manifest --out data/manifest.jsonl
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bpm as bpm_mod
from . import probe as probe_mod
from .ableton import CLASS_TAGS, Library, b64

SCHEMA = 1
CACHE = Path("cache/probe.jsonl")
_MISS = object()


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run killed mid-write truncates the last line
            if r["v"] is not None:  # failures are retried, not remembered
                out[r["k"]] = r["v"]
    return out


def _cache_key(path: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{path}:{st.st_size}:{int(st.st_mtime)}"


def build(out_path: Path, *, limit: int | None, workers: int, no_embedding: bool):
    lib = Library()
    emb = {} if no_embedding else lib.embeddings()
    fams = lib.families()
    one_shots, loops = lib.kind_members("one_shot"), lib.kind_members("loop")

    rows = {}
    for cls in CLASS_TAGS:
        pack, auto = lib.class_members(cls)
        for fid in pack | auto:
            src = "both" if fid in pack and fid in auto else ("pack" if fid in pack else "auto")
            kind = "one_shot" if fid in one_shots else "loop" if fid in loops else "other"
            # A file carrying two class tags is vanishingly rare (5 library-wide);
            # first writer wins rather than silently duplicating the row.
            rows.setdefault(fid, {"class": cls, "label_source": src, "kind": kind})
    # Unclassified loops still matter: they are background and evaluation material.
    for fid in loops:
        rows.setdefault(fid, {"class": None, "label_source": None, "kind": "loop"})

    ids = sorted(rows)[: limit or None]
    cache = _load_cache(CACHE)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fresh = []

    def work(fid):
        path = lib.path_of(fid)
        key = _cache_key(path)
        if key is None:
            return fid, None, None
        hit = cache.get(key, _MISS)
        if hit is not _MISS:
            return fid, hit, None
        p = probe_mod.probe(path, onset=rows[fid]["kind"] == "one_shot")
        d = p.asdict() if p else None
        return fid, d, ((key, d) if d is not None else None)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(workers) as ex, out_path.open("w") as out, CACHE.open("a") as cf:
        results = []
        for fid, d, new in ex.map(work, ids):
            if new:
                cf.write(json.dumps({"k": new[0], "v": new[1]}) + "\n")
            results.append((fid, d))
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(ids)}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

        counts = {"missing": 0, "unreadable": 0, "discarded": 0, "with_bpm": 0}
        body = []
        for fid, d in results:
            r = dict(rows[fid])
            r["file_id"] = fid
            r["name"] = lib.name_of(fid)
            r["path"] = lib.path_of(fid)
            if d is None:
                # Kept as a row, not dropped: most of these are Ableton's
                # compressed AIFC, which no decoder here reads, and they are
                # concentrated in the pack-tagged labels.
                missing = _cache_key(r["path"]) is None
                counts["missing" if missing else "unreadable"] += 1
                r["readable"] = False
                r["discard"] = True
                r["discard_reason"] = "file_missing" if missing else "undecodable"
                body.append(r)
                continue
            r["readable"] = True
            r.update(d)
            if r["kind"] == "loop":
                t = bpm_mod.tempo_of(r["name"], lib.folders_of(fid), r["duration_s"])
                r["bpm"] = t.bpm if t else None
                r["bpm_source"] = t.source if t else None
                r["bars"] = t.bars if t else None
                counts["with_bpm"] += bool(t)
            if r.get("discard"):
                counts["discarded"] += 1
            r["families"] = sorted(fams.get(fid, ()))
            if fid in emb:
                r["embedding"] = b64(emb[fid])
            body.append(r)

        header = {
            "schema": SCHEMA,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "index": lib.fingerprint(),
            "params": {
                "silence_dbfs": probe_mod.SILENCE_DBFS,
                "attack_rel_db": probe_mod.ATTACK_REL_DB,
                "attack_discard_ms": probe_mod.ATTACK_DISCARD_MS,
                "bpm_tol": bpm_mod.TOL,
                "bpm_bar_counts": list(bpm_mod.BAR_COUNTS),
            },
            "counts": {"rows": len(body), **counts},
        }
        out.write(json.dumps(header) + "\n")
        for r in body:
            out.write(json.dumps(r) + "\n")
    return header


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-embedding", action="store_true")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    h = build(a.out, limit=a.limit, workers=a.workers, no_embedding=a.no_embedding)
    print(json.dumps(h["counts"], indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
