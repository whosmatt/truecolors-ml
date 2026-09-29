"""Five-fold song split for cross-validation: grouped by artist, balanced by genre.

A hash split put 8 labelled songs in test (2026-09-28), too few to separate
models. Folds test every labelled song once. Songs by one artist share a fold,
since they share production and kits; splitting a song itself would test on
its own tempo and samples. Balance is over labelled songs (unlabelled ones
follow their artist), so the assignment can shift as labelling goes on: each
experiment keeps a copy of the folds it used.

    python -m dataset.songfolds   # -> songs/folds.json {song id: fold}
"""

import json
from collections import Counter, defaultdict

from .songgrid import SONGS

FOLDS = 5
VAL_EVERY = 8  # every 8th training-fold song (by id order) is validation


def artist(row: dict) -> str:
    return row["artist"].split(";")[0].split(",")[0].strip().lower()


def genre(row: dict) -> str:
    return (row.get("genres") or "none").split(",")[0].strip() or "none"


def assign(index: dict, labelled: set) -> dict:
    groups = defaultdict(list)
    for k, r in index.items():
        if r.get("status") == "ok":
            groups[artist(r)].append(k)
    size = [0] * FOLDS
    per_genre = [Counter() for _ in range(FOLDS)]
    out = {}
    n_lab = lambda ids: sum(k in labelled for k in ids)
    for a, ids in sorted(groups.items(), key=lambda kv: (-n_lab(kv[1]), -len(kv[1]), kv[0])):
        lab = [k for k in ids if k in labelled]
        g = Counter(genre(index[k]) for k in lab)
        f = min(range(FOLDS), key=lambda i: (sum(per_genre[i][x] * n for x, n in g.items()), size[i], i))
        for k in ids:
            out[k] = f
        size[f] += len(lab)
        per_genre[f].update(g)
    return out


def load() -> dict:
    return json.loads((SONGS / "folds.json").read_text())


def split_for(song_id: str, fold: int, folds: dict) -> str:
    """Fold -1: no test songs, for a final model (the CV measured held-out performance)."""
    f = folds[song_id]
    if f == fold:
        return "test"
    train_ids = sorted(k for k, v in folds.items() if v != fold)
    return "val" if train_ids.index(song_id) % VAL_EVERY == 0 else "train"


def main():
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    labelled = {p.stem for p in (SONGS / "labels").glob("*.json")}
    folds = assign(index, labelled)
    (SONGS / "folds.json").write_text(json.dumps(folds, indent=0))
    for f in range(FOLDS):
        ids = [k for k, v in folds.items() if v == f]
        lab = [k for k in ids if k in labelled]
        print(f"fold {f}: {len(ids)} songs, {len(lab)} labelled, "
              f"{len({artist(index[k]) for k in ids})} artists, "
              f"genres {Counter(genre(index[k]) for k in lab).most_common(4)}")


if __name__ == "__main__":
    main()
