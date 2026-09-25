"""Results pages for approach 6 (context shape, pooling, sequence models) and
approach 7 (real loops with a file-start grid, melodic loops).

Scores come from `train.compare` (equal 461-block window on every test set),
`train.ablate` and `train.loopstart`, cached next to each page.

    python -m train.report6
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import data
from .report import GRID, INK, INK2, SERIES, new_fig, style

OUT6 = Path("results/6-context-and-sequence")
OUT7 = Path("results/7-loop-grids")
BLOCK_S = 512 / 48000

CONTEXT = [
    ("control: fine 16, mid 16x4, coarse 12x16", "i_ctrl"),
    ("no coarse", "i_nocoarse"),
    ("no mid", "i_nomid"),
    ("coarse 24x16, 4.9 s", "i_coarse24"),
    ("mid 8x4, coarse 20x16, 3.9 s", "i_redist"),
    ("mid 24x4", "i_mid24"),
    ("fine 8, mid 24x4", "i_fine5mid24"),
]
POOLING = [
    ("mean (control)", "i_ctrl"),
    ("coarse max", "i_cmax"),
    ("max", "i_max"),
    ("mean + max", "i_meanmax"),
    ("max + peak position", "i_maxpos"),
]
SEQUENCE = [
    ("window MLP 528-128-64 (control)", "i_ctrl"),
    ("causal TCN, 32 channels", "j_tcn32"),
    ("causal TCN, 48 channels", "j_tcn48"),
    ("GRU 64", "j_gru64"),
    ("GRU 128", "j_gru128"),
]
LOOPS = [
    ("rendered clips only (control)", "i_ctrl"),
    ("+ melodic, phase-free loss, weight 0.3", "k_mil03"),
    ("+ melodic, phase-free loss, weight 1", "k_mil1"),
    ("+ melodic, phase-free loss, weight 3", "k_mil3"),
    ("+ drum loops, file-start grid", "l_drum"),
    ("+ drum and melodic loops, file-start grid", "l_drummel"),
    ("+ drum loops, file-start grid; melodic, phase-free 0.3", "l_drum_mil"),
]

CANDIDATE = [
    ("music v2, delivered (g_music2)", "g_music2"),
    ("+ drum loops, file-start grid; melodic, phase-free 0.3", "m_cand"),
]
# python -m train.musiceval runs/g_music2_s0 runs/m_cand_s0 runs/m_cand_s1
#   --clips data/features4 --loops data/loops4   (2026-09-24)
GATE = ["| music v2, seed 0 | 88.3% | 86.7% | 100.0% | 79.4% |",
        "| candidate, 2 seeds | 88.3% ±0.2 | 86.9% ±0.2 | 100.0% ±0.0 | 80.0% ±0.3 |"]

MERMAID6 = """```mermaid
flowchart LR
  F["12 features<br/>93.75 blocks/s"] --> W & S
  subgraph W["window MLP, tested"]
    T["tiers: fine / mid / coarse<br/>shape and pooling varied"] --> MLP["MLP 128-64"]
  end
  subgraph S["sequence models, tested"]
    TCN["causal dilated TCN<br/>d = 1..128, RF 513 blocks"]
    GRU["GRU, state carried"]
  end
  W --> AC["stage 2: autocorrelation"]
  S --> AC
  AC --> G["BPM + anchor"]
```"""

MERMAID7 = """```mermaid
flowchart LR
  CLIPS["rendered .alc clips<br/>exact grid"] --> M
  DRUM["real drum loops<br/>grid from file start"] --> M
  MEL["melodic loops<br/>grid from file start, or<br/>tempo only: phase-free loss"] --> M
  M["MLP 528-128-64<br/>beat, offset, drum"] --> AC["stage 2"] --> G["BPM + anchor"]
```"""


def load(path):
    return json.loads(path.read_text())


def rows(board, key):
    return [v for k, v in board.items() if k.rsplit("/", 1)[-1].rsplit("_s", 1)[0] == key]


def vals(board, key, test, field, scale=100.0):
    return np.array([r[test][field] for r in rows(board, key) if test in r]) * scale


def fmt(v, unit="%", nd=1):
    if len(v) == 0:
        return "-"
    s = f"{v.mean():.{nd}f}{unit}"
    return s + (f" ±{v.std():.{nd}f}" if len(v) > 1 else "")


def table(board, configs, cols):
    head = "| configuration | " + " | ".join(c[0] for c in cols) + " | MACs | seeds |"
    L = [head, "|---|" + "---|" * (len(cols) + 2)]
    for lbl, key in configs:
        rs = rows(board, key)
        cells = [fmt(vals(board, key, t, f, 1.0 if u else 100.0), u or "%", nd)
                 for _, t, f, u, nd in cols]
        L.append(f"| {lbl} | " + " | ".join(cells) + f" | {rs[0]['macs']:,} | {len(rs)} |")
    return L


RENDERED = [("rendered: usable", "clips", "usable", "", 1),
            ("rendered: tempo +octave", "clips", "octave", "", 1),
            ("real drum loops: tempo", "loops", "tempo", "", 1),
            ("melodic loops: tempo", "melodic", "tempo", "", 1)]
REAL = [("rendered: usable", "clips", "usable", "", 1),
        ("real drum: usable", "loops_grid", "usable", "", 1),
        ("real drum: phase", "loops_grid", "phase_exact", " ms", 1),
        ("real drum: tempo", "loops", "tempo", "", 1),
        ("melodic: usable", "melodic_grid", "usable", "", 1),
        ("melodic: phase", "melodic_grid", "phase_exact", " ms", 1),
        ("melodic: tempo", "melodic", "tempo", "", 1)]


def corpus_rows(dirs):
    out = []
    for name, path in dirs:
        meta = data.meta(Path(path))
        for s in ("train", "val", "test"):
            sp = meta["splits"][s]
            size = (Path(path) / f"{s}.npz").stat().st_size / 2**20
            out.append(f"| {name} | {s} | {sp['takes']:,} | {sp['blocks']:,} | "
                       f"{sp['blocks'] * BLOCK_S / 3600:.2f} | {size:,.0f} MB |")
    return out


def dots(board, configs, panels, path, ref=0):
    """One panel per metric, one row per configuration; dots are seeds."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 1.6, 0.34 * len(configs) + 1.1),
                             dpi=160, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    axes = np.atleast_1d(axes)
    y = np.arange(len(configs))[::-1]
    for ax, (title, test, field) in zip(axes, panels):
        for yi, (_, key) in zip(y, configs):
            v = vals(board, key, test, field)
            if not len(v):
                continue
            ax.scatter(v, [yi] * len(v), s=16, color=SERIES["kick"], zorder=3,
                       edgecolors="white", linewidths=1.0)
            ax.plot([v.mean()] * 2, [yi - 0.28, yi + 0.28], color=INK, linewidth=2, zorder=4)
        c = vals(board, configs[ref][1], test, field).mean()
        ax.axvline(c, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        style(ax, xlabel="% of test takes", title=title)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_yticks(y, [c[0] for c in configs], fontsize=7)
    axes[0].set_ylim(-0.7, len(configs) - 0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def page6():
    board = load(OUT6 / "compare.json")
    abl = load(OUT6 / "ablation.json")
    L = ["# 6: context shape, pooling, sequence models", "", MERMAID6, "",
         "## Configuration", "", "| | |", "|---|---|",
         "| front end | `FE_SPEC_VERSION 1`, variant `hicut` (2) |",
         "| training | rendered clips, 2026-09-20 index (`data/features3`); beat, offset, drum heads |",
         "| window models | `train/grid5.py`, flags `--mid --coarse --fine-past --mid-pool --coarse-pool` |",
         "| sequence models | `train/seq.py`: whole takes, causal, 2-block lookahead as the MLP |",
         "| scoring | `train/compare.py`: blocks from 461 into each take, every model (the longest lookback tested) |",
         "",
         "## Inference-time ablation, control model, 269-block window", "",
         "| frames replaced | blank: usable | blank: tempo +octave | shuffle: usable |",
         "|---|---|---|---|"]
    keys = list(next(iter(abl.values())).keys())
    for k in keys:
        if k.endswith(": shuffle"):
            continue
        name = k.replace(": blank", "")
        u = np.array([abl[r][k]["usable"] for r in abl]) * 100
        o = np.array([abl[r][k]["octave"] for r in abl]) * 100
        sh = (np.array([abl[r][f"{name}: shuffle"]["usable"] for r in abl]) * 100
              if name != "none" else u)
        L.append(f"| {name} | {fmt(u)} | {fmt(o)} | {fmt(sh)} |")
    L += ["", "## Context shape, retrained", "", *table(board, CONTEXT, RENDERED), "",
          "## Pooling of mid and coarse tiers, retrained", "", *table(board, POOLING, RENDERED), "",
          "## Sequence models", "", *table(board, SEQUENCE, RENDERED + [
              ("rendered: phase, octave error", "clips", "phase_octave", " ms", 1)]), "",
          "![context](context.png)", ""]
    (OUT6 / "README.md").write_text("\n".join(L) + "\n")
    dots(board, CONTEXT + POOLING[1:] + SEQUENCE[1:],
         [("rendered clips: usable grid", "clips", "usable"),
          ("real drum loops: tempo", "loops", "tempo")], OUT6 / "context.png")


def page7():
    board = load(OUT7 / "compare.json")
    ls = load(OUT7 / "loopstart.json")
    L = ["# 7: real loops with a file-start grid", "", MERMAID7, "",
         "## Configuration", "", "| | |", "|---|---|",
         "| front end | `FE_SPEC_VERSION 1`, variant `hicut` (2); loop corpora built at v2, bit-identical |",
         "| model | window MLP 528-128-64, beat + offset + drum heads, 76,160 MACs |",
         "| drum and melodic grid | `dataset/loopset.py --grid --max-len-err 5e-4`: beats at k*60/bpm from the file start |",
         "| melodic loops | no Drums tag, BPM verified, backing-bed packs excluded |",
         "| phase-free loss | per 512-block segment, BCE against the grid at every 1/3-block phase of the known period; the best phase counts; +-1 block tolerance |",
         "| scoring | `train/compare.py`, 461-block window |",
         "| provenance | manifest `b38ec7dd9e96` (2026-09-24); IR `5279ea9dd9f4` |",
         "", "## Dataset", "",
         "| corpus | split | takes | blocks | hours | size |", "|---|---|---|---|---|---|",
         *corpus_rows([("rendered clips, 2026-09-20", "data/features3"),
                       ("real drum loops, file-start grid", "data/loops_grid"),
                       ("melodic loops, file-start grid", "data/melodic_grid"),
                       ("melodic loops, tempo only", "data/melodic")]), "",
         "## Do loops start on a beat? Models trained on rendered clips only, test split", "",
         "| corpus | model | tempo-exact takes | grid within 25 ms of file start | median | half a beat off |",
         "|---|---|---|---|---|---|"]
    for k, v in ls.items():
        corpus, run = k.split(":", 1)
        e = np.array(v["eighths"])
        L.append(f"| {corpus} | `{run}` | {v['tempo_exact']} / {v['takes']} | {v['within_25ms']*100:.1f}% | "
                 f"{v['median_ms']:.1f} ms | {e[4] / e.sum() * 100:.1f}% |")
    L += ["", "## Training on loops, rendered clips 2026-09-20", "", *table(board, LOOPS, REAL), "",
          "![loops](loops.png)", "",
          "## Full configuration, rendered clips 2026-09-24 (945 test takes), music head on", "",
          *table(load(OUT7 / "candidate.json"), CANDIDATE, REAL), "",
          "## Music gate, per-take median, val split, threshold 0.7", "",
          "| model | real loops kept | non-music rejected | silence rejected | rendered kept |",
          "|---|---|---|---|---|",
          *GATE, ""]
    (OUT7 / "README.md").write_text("\n".join(L) + "\n")
    dots(board, LOOPS, [("rendered clips: usable", "clips", "usable"),
                        ("real drum loops: usable", "loops_grid", "usable"),
                        ("melodic loops: tempo", "melodic", "tempo")], OUT7 / "loops.png")


def main():
    page6()
    page7()
    print(f"-> {OUT6}/README.md, {OUT7}/README.md")


if __name__ == "__main__":
    main()
