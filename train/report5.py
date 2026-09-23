"""Results page for approach 5: multi-task heads on the long-context model.

    python -m train.report5
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import matplotlib.pyplot as plt
import numpy as np

from . import data, mres, musiceval
from .report import GRID, INK, INK2, SERIES, new_fig, style

OUT = Path("results/5-multitask-heads")
BLOCK_S = 512 / 48000

# (label, scoreboard keys); old pipeline, clips test split
HEADS = [
    ("beat + offset + drum (control)", ["e_base", "e_base_s1"]),
    ("+ music", ["e_music", "e_music_s1"]),
    ("- drum", ["e_noaux", "e_noaux_s1"]),
    ("+ phrase", ["e_phrase_only"]),
    ("+ downbeat", ["e_downbeat_only"]),
    ("+ downbeat, gentle weight", ["e_down_soft"]),
    ("+ downbeat, trunk 192-96", ["e_down_wide"]),
    ("+ downbeat + phrase", ["e_downbeat"]),
]
DOWNBEAT_F1 = {"+ downbeat": 0.378, "+ downbeat, gentle weight": 0.310}

FRONTEND = [("hi-cut 4 kHz (variant 2)", ["hicut-s0", "hicut-s1"]),
            ("open (variant 0)", ["open-s0", "open-s1"])]

PIPELINE = [("numpy batches, eager step", ["e_music", "e_music_s1"]),
            ("GPU-resident, XLA step", ["fast-music", "fast-music-s1", "fast-music-s2"])]

# (label, runs, scoreboard keys)
GATE = [
    ("v1: music head", ["runs/e_music", "runs/e_music_s1"], ["e_music", "e_music_s1"]),
    ("+ silence is not music", ["runs/f_music_sil"], ["music-silence"]),
    ("+ real loops as music", ["runs/f_music_loops"], ["music-loops"]),
    ("+ both", ["runs/f_music_both", "runs/f_music_both_s1"], ["music-both", "music-both-s1"]),
    ("+ both, music weight 0.3", ["runs/f_mboth_p03", "runs/f_mboth_p03_s1"],
     ["mboth-p0.3", "mboth-p0.3-s1"]),
]
GATE_T = 0.7

CORPUS = [("2026-09-20 index, 985 clips", ["olddb-s0@new", "olddb-s1@new"],
           ["mboth-p0.3", "mboth-p0.3-s1"]),
          ("2026-09-24 index, 1,123 clips", ["newdb-s0@new", "newdb-s1@new"],
           ["newdb-s0@old", "newdb-s1@old"])]

MERMAID = """```mermaid
flowchart LR
  CLIPS["rendered .alc clips<br/>exact grid"] --> MERGE
  NOISE["non-music takes<br/>speech, foley, atmosphere, fx, silence"] --> MERGE
  LOOPS["real drum loops<br/>music only, grid masked"] --> MERGE
  MERGE["masked multi-task corpus"] --> C["multi-resolution context<br/>2.9 s, 528 inputs"] --> M
  subgraph M["stage 1: MLP 528-128-64"]
    B["beat"]
    O["beat offset"]
    AUX["drum, auxiliary"]
    MU["music"]
    DB["downbeat, tested"]
    PH["phrase, tested"]
  end
  B --> AC["stage 2: autocorrelation"] --> G["BPM + anchor"]
  O --> AC
  MU --> GATE["music gate"]
```"""


def agg(board, keys, field, scale=100.0):
    v = np.array([board[k]["clips" if field in ("usable", "octave", "phase_exact") else "loops"][field]
                  for k in keys]) * scale
    return v


def fmt(v, unit="%", nd=1):
    s = f"{v.mean():.{nd}f}{unit}"
    return s + (f" ±{v.std():.{nd}f}" if len(v) > 1 else "")


def grid_row(board, keys):
    u = agg(board, keys, "usable")
    o = agg(board, keys, "octave")
    p = agg(board, keys, "phase_exact", 1.0)
    t = agg(board, keys, "tempo")
    return u, f"{fmt(u)} | {fmt(o)} | {fmt(p, ' ms', 2)} | {fmt(t)}"


def corpus_rows():
    rows = []
    for name, path in (("rendered clips, 2026-09-20", "data/features3"),
                       ("rendered clips, 2026-09-24", "data/features4"),
                       ("non-music", "data/noise"), ("real drum loops", "data/loops")):
        meta = data.meta(Path(path))
        for s in ("train", "val", "test"):
            sp = meta["splits"][s]
            size = (Path(path) / f"{s}.npz").stat().st_size / 2**20
            rows.append(f"| {name} | {s} | {sp['takes']:,} | {sp['blocks']:,} | "
                        f"{sp['blocks'] * BLOCK_S / 3600:.2f} | {size:,.0f} MB |")
    return rows


def gate_scores(runs, clips, loops, noise, spec):
    """-> {group: [% correct side per run]} at GATE_T."""
    out = {}
    for run in runs:
        nz = np.load(Path(run) / "norm.npz")
        m = keras.models.load_model(Path(run) / "model.keras", compile=False)
        rng = np.random.default_rng(0)
        for n, s in (("rendered kept", clips), ("real loops kept", loops), ("noise", noise)):
            med, lvl = musiceval.take_medians(m, s, nz["mean"], nz["scale"], spec, rng)
            if n == "noise":
                groups = {"non-music rejected": (med[lvl >= musiceval.SILENT_DB], False),
                          "silence rejected": (med[lvl < musiceval.SILENT_DB], False)}
            else:
                groups = {n: (med, True)}
            for g, (v, pos) in groups.items():
                out.setdefault(g, []).append(
                    float(np.mean(v >= GATE_T if pos else v < GATE_T) * 100))
    return out


GROUPS = ["real loops kept", "non-music rejected", "silence rejected", "rendered kept"]


def plot_gate(gate, path):
    first, last = gate[GATE[0][0]], gate["v2: + both, weight 0.3, 2026-09-24 index"]
    fig, ax = new_fig(6.4, 3.0)
    x = np.arange(len(GROUPS))
    w = 0.36
    for k, (lbl, g, col) in enumerate((("v1", first, SERIES["kick"]),
                                       ("v2", last, SERIES["snare"]))):
        vals = [np.mean(g[n]) for n in GROUPS]
        ax.bar(x + (k - 0.5) * (w + 0.02), vals, width=w, color=col, label=lbl, zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * (w + 0.02), v + 1.5, f"{v:.0f}", ha="center",
                    fontsize=7, color=INK2)
    ax.set_xticks(x, GROUPS)
    ax.set_ylim(0, 110)
    style(ax, ylabel=f"% of takes, threshold {GATE_T}")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heads(board, path):
    fig, ax = new_fig(6.4, 3.0)
    labels = [h[0] for h in HEADS]
    y = np.arange(len(labels))[::-1]
    # Dots, not bars: the axis starts at 55, and a bar's length would lie.
    for yi, (lbl, keys) in zip(y, HEADS):
        u = agg(board, keys, "usable")
        ax.scatter(u, [yi] * len(u), s=18, color=SERIES["kick"], zorder=3,
                   edgecolors="white", linewidths=1.0)
        ax.plot([u.mean()] * 2, [yi - 0.28, yi + 0.28], color=INK, linewidth=2, zorder=4)
        ax.text(75.6, yi, f"{u.mean():.1f}", va="center", ha="left", fontsize=7, color=INK2)
    ctrl = agg(board, HEADS[0][1], "usable").mean()
    ax.axvline(ctrl, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(55, 75)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    style(ax, xlabel="usable grid, % of test takes (dots: seeds, bar: mean)")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    board = json.loads(Path("results/scoreboard.json").read_text())
    cache = OUT / "final.json"
    spec = mres.Spec()

    if cache.exists() and not a.recompute:
        gate = json.loads(cache.read_text())["gate"]
    else:
        noise = data.load(Path("data/noise"), "val")
        old = (data.load(Path("data/features3"), "val"), data.load(Path("data/loops"), "val"))
        new = (data.load(Path("data/features4"), "val"), data.load(Path("data/loops4"), "val"))
        gate = {lbl: gate_scores(runs, *old, noise, spec) for lbl, runs, _ in GATE}
        gate["v2: + both, weight 0.3, 2026-09-24 index"] = gate_scores(
            ["runs/g_music2_s0", "runs/g_music2_s1"], *new, noise, spec)
        cache.write_text(json.dumps({"gate": gate}, indent=1))

    L = ["# 5: multi-task heads", "", MERMAID, "", "## Configuration", "",
         "| | |", "|---|---|",
         "| front end | `FE_SPEC_VERSION 1`, variant `hicut` (2) |",
         "| context | 2.87 s, 44 frames, 528 inputs |",
         "| model | [128, 64] hidden, heads selectable (`train/grid5.py`) |",
         "| size | 76,224 MAC/inference, 86,048 B int8 with music head |",
         "| music labels | silence: past 683 ms mean spl < 52 dB is 0; real loops 1; positive weight 0.3 |",
         "| provenance | manifest `a9830eda7ba1` (2026-09-20), `b38ec7dd9e96` (2026-09-24); IR `5279ea9dd9f4` |",
         "", "## Dataset", "",
         "| corpus | split | takes | blocks | hours | size |", "|---|---|---|---|---|---|",
         *corpus_rows(), "",
         "## Heads: rendered clips, test split", "",
         "| heads | usable grid | tempo ok (+octave) | phase, tempo exact | real loops: tempo | downbeat F1 | MACs |",
         "|---|---|---|---|---|---|---|"]
    for lbl, keys in HEADS:
        _, row = grid_row(board, keys)
        f1 = f"{DOWNBEAT_F1[lbl]:.3f}" if lbl in DOWNBEAT_F1 else ""
        L.append(f"| {lbl} | {row} | {f1} | {board[keys[0]]['macs']:,} |")
    L += ["", "![heads](heads.png)", "",
          f"## Music gate: per-take median, val split, threshold {GATE_T}", "",
          "| music head | " + " | ".join(GROUPS) + " | usable grid |",
          "|---|" + "---|" * (len(GROUPS) + 1)]
    for lbl, _, keys in GATE:
        g = gate[lbl]
        u = agg(board, keys, "usable")
        L.append(f"| {lbl} | " + " | ".join(fmt(np.array(g[n])) for n in GROUPS) + f" | {fmt(u)} |")
    g = gate["v2: + both, weight 0.3, 2026-09-24 index"]
    u = agg(board, CORPUS[1][2], "usable")
    L.append("| v2: + both, weight 0.3, 2026-09-24 index | "
             + " | ".join(fmt(np.array(g[n])) for n in GROUPS) + f" | {fmt(u)} |")
    L += ["", "![gate](gate.png)", "",
          "## Front end: hi-cut, 2 seeds, 985 clips each", "",
          "| front end | usable grid | tempo ok (+octave) | phase, tempo exact | real loops: tempo |",
          "|---|---|---|---|---|"]
    for lbl, keys in FRONTEND:
        L.append(f"| {lbl} | {grid_row(board, keys)[1]} |")
    L += ["", "## Corpus: music v2, 2 seeds", "",
          "| training index | usable, 2026-09-20 test (828 takes) | usable, 2026-09-24 test (945 takes) |",
          "|---|---|---|"]
    for lbl, new_keys, old_keys in CORPUS:
        L.append(f"| {lbl} | {fmt(agg(board, old_keys, 'usable'))} | "
                 f"{fmt(agg(board, new_keys, 'usable'))} |")
    L += ["", "## Training pipeline, same heads (beat, offset, drum, music)", "",
          "| pipeline | usable grid | phase, tempo exact | minutes per run |", "|---|---|---|---|"]
    for lbl, keys in PIPELINE:
        mins = [json.loads(Path(board[k]["run"], "result.json").read_text())["minutes"] for k in keys]
        L.append(f"| {lbl} | {fmt(agg(board, keys, 'usable'))} | "
                 f"{fmt(agg(board, keys, 'phase_exact', 1.0), ' ms', 2)} | {np.mean(mins):.1f} |")
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    plot_heads(board, OUT / "heads.png")
    plot_gate(gate, OUT / "gate.png")
    print(f"-> {OUT}/README.md")


if __name__ == "__main__":
    main()
