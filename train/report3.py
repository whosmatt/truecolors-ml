"""Results page for approach 3, with the full cross-approach comparison.

    python -m train.report3
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import numpy as np

from . import data, evaluate, gridmetrics, mres
from .grid import activation as act_plain
from .grid3 import activation as act_mres
from .report import GRID, INK, INK2, SERIES, SURFACE, dataset_stats, new_fig, style

APPROACHES = [
    ("raw flux", None, None, 0),
    ("1: drum detector", ["runs/none", "runs/none_s1", "runs/none_s2"], "hit", 33216),
    ("2: beat activation", ["runs/grid", "runs/grid_s1", "runs/grid_s2"], "beat", 33088),
    ("3: long context", ["runs/grid3", "runs/grid3_s1", "runs/grid3_s2"], "beat", None),
]
COLOR = {"raw flux": "#9aa0a6", "1: drum detector": SERIES["kick"],
         "2: beat activation": SERIES["snare"], "3: long context": SERIES["hihat"]}

MERMAID = """```mermaid
flowchart LR
  F["12 features<br/>93.75 blocks/s"] --> C
  subgraph C["multi-resolution context: 2.9 s"]
    FINE["fine: 16 frames<br/>full rate, 171 ms"]
    MID["mid: 16 frames<br/>4-block means, 683 ms"]
    COARSE["coarse: 12 frames<br/>16-block means, 2048 ms"]
  end
  C --> M
  subgraph M["stage 1: MLP 528-128-64"]
    B["beat head"]
    O["beat offset head"]
    AUX["drum heads, auxiliary"]
  end
  M --> ACT["beat activation"] --> AC["stage 2: autocorrelation<br/>unchanged"] --> G["BPM + anchor"]
```
"""


def collect(te):
    spec = mres.Spec()
    out = {}
    for name, runs, key, _ in APPROACHES:
        if runs is None:
            flux = np.maximum(te.X[:, 5], te.X[:, 9])
            out[name] = [gridmetrics.per_take(flux, te)]
            continue
        rows = []
        for r in runs:
            if not Path(r, "model.keras").exists():
                continue
            m = keras.models.load_model(Path(r) / "model.keras", compile=False)
            nz = np.load(Path(r) / "norm.npz")
            a = (act_mres(m, te, nz["mean"], nz["scale"], spec, key=key)
                 if name.startswith("3") else
                 act_plain(m, te, nz["mean"], nz["scale"], key=key))
            rows.append(gridmetrics.per_take(a, te))
        out[name] = rows
    return out


def fig_grid(res, path):
    import matplotlib.pyplot as plt
    fig, ax = new_fig(5.8, 3.3)
    names = list(res)
    w = 0.38
    for k, (metric, label) in enumerate((("usable", "usable grid"), ("octave", "tempo ok (+octave)"))):
        for i, n in enumerate(names):
            v = np.array([r[metric] for r in res[n]]) * 100
            ax.bar(i + (k - 0.5) * w, v.mean(), width=w * 0.92, color=COLOR[n],
                   alpha=1.0 if k == 0 else 0.42, edgecolor=SURFACE, linewidth=1.5,
                   label=label if i == 0 else None)
            if len(v) > 1:
                ax.errorbar(i + (k - 0.5) * w, v.mean(), yerr=v.std(), color=INK2,
                            linewidth=1.2, capsize=3)
            ax.text(i + (k - 0.5) * w, v.mean() + 2.2, f"{v.mean():.1f}",
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(0, 100)
    style(ax, "", "% of takes", "End to end, rendered clips (exact grid)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_phase(res, path):
    import matplotlib.pyplot as plt
    fig, ax = new_fig(5.8, 3.3)
    names = list(res)
    w = 0.38
    for k, (metric, label) in enumerate((("phase_exact", "tempo exact"),
                                         ("phase_octave", "octave error"))):
        for i, n in enumerate(names):
            v = np.array([r[metric] for r in res[n]])
            ax.bar(i + (k - 0.5) * w, np.nanmean(v), width=w * 0.92, color=COLOR[n],
                   alpha=1.0 if k == 0 else 0.42, edgecolor=SURFACE, linewidth=1.5,
                   label=label if i == 0 else None)
            ax.text(i + (k - 0.5) * w, np.nanmean(v) + 4, f"{np.nanmean(v):.0f}",
                    ha="center", fontsize=7.5, color=INK2)
    ax.axhline(gridmetrics.PHASE_OK_MS, color=GRID, linewidth=1.0, linestyle="--")
    ax.text(len(names) - 0.5, gridmetrics.PHASE_OK_MS + 3, "usable threshold",
            fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(range(len(names)), names, fontsize=8)
    style(ax, "", "median grid phase error (ms)", "Phase, split by whether the tempo was exact")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/3-long-context"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    te = data.load(a.features, "test")
    res = collect(te)
    fig_grid(res, a.out / "grid.png")
    fig_phase(res, a.out / "phase.png")

    run = json.loads(Path("runs/grid3/result.json").read_text())
    stats = dataset_stats(a.features, a.manifest)
    meta, sp = stats["meta"], stats["splits"]
    mb = lambda b: f"{b/1024/1024:.0f} MB"
    spec = run["spec"]

    L = ["# 3: long context\n", MERMAID, "\n## Configuration\n", "| | |", "|---|---|",
         f"| front end | `FE_SPEC_VERSION {meta['fe_spec_version']}`, variant `{meta.get('fe_variant_name')}` |",
         f"| context | {run['context_s']:.2f} s, {run['frames']} frames, {run['frames']*12} inputs |",
         f"| fine | {spec['fine_past']} past + {spec['fine_future']} future, full rate |",
         f"| mid | {spec['mid_frames']} frames of {spec['mid_stride']} blocks |",
         f"| coarse | {spec['coarse_frames']} frames of {spec['coarse_stride']} blocks |",
         f"| model | {run['hidden']} hidden, beat + offset heads, drum heads auxiliary |",
         f"| size | {run['macs']:,} MAC/inference, ~{run['macs']/1024:.0f} KB int8 |",
         f"| provenance | manifest `{meta['manifest_sha256'][:12]}`, IR `{(meta.get('ir_sha256') or '-')[:12]}` |\n",
         "## Dataset\n", "| split | takes | blocks | hours | size | beats |",
         "|---|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        if s in sp:
            d = np.load(a.features / f"{s}.npz")
            L.append(f"| {s} | {sp[s]['takes']:,} | {sp[s]['blocks']:,} | {sp[s]['hours']:.2f} "
                     f"| {mb(sp[s]['bytes'])} | {int(d['beat'].sum()):,} |")
    L += [f"| **total** | {sum(v['takes'] for v in sp.values()):,} | "
          f"{sum(v['blocks'] for v in sp.values()):,} | {sum(v['hours'] for v in sp.values()):.2f} "
          f"| {mb(stats['features_bytes'])} | |\n",
          "## End to end: rendered clips, exact grid (identical stage 2, 3 seeds)\n",
          "| approach | usable grid | tempo ok (+octave) | phase, tempo exact | phase, octave error | MACs |",
          "|---|---|---|---|---|---|"]
    for name, runs, _, macs in APPROACHES:
        r = res[name]
        u = np.array([x["usable"] for x in r]) * 100
        o = np.array([x["octave"] for x in r]) * 100
        pe = np.nanmean([x["phase_exact"] for x in r])
        po = np.nanmean([x["phase_octave"] for x in r])
        sd = f" ±{u.std():.1f}" if len(u) > 1 else ""
        sdo = f" ±{o.std():.1f}" if len(o) > 1 else ""
        mc = run["macs"] if macs is None else macs
        L.append(f"| {name} | {u.mean():.1f}%{sd} | {o.mean():.1f}%{sdo} | {pe:.1f} ms "
                 f"| {po:.1f} ms | {mc:,} |")
    final = Path("results/3-long-context/final.json")
    if final.exists():
        f = json.loads(final.read_text())
        n_loops = f["3: long context"]["loops"]["n"]
        L += ["", f"## Tempo prior: measured from training clip tempos, peak 120 BPM "
              f"({n_loops} real loops, tempo only)\n",
              "| approach | rendered: usable | + prior | real loops: tempo | + prior "
              "| real loops: +octave | + prior |",
              "|---|---|---|---|---|---|---|"]
        for name, _, _, _ in APPROACHES:
            r = f[name]
            L.append(f"| {name} | {r['rendered']['usable']:.1%} | {r['rendered + prior']['usable']:.1%} "
                     f"| {r['loops']['exact']:.1%} | {r['loops + prior']['exact']:.1%} "
                     f"| {r['loops']['octave']:.1%} | {r['loops + prior']['octave']:.1%} |")
        L += [""]
    L += ["", "![end to end](grid.png)", "", "![phase](phase.png)", ""]
    (a.out / "README.md").write_text("\n".join(L))
    print(f"-> {a.out}")
    for p in sorted(a.out.iterdir()):
        print(f"   {p.name:14} {p.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
