"""Results page for approach 2 (beat activation).

    python -m train.report_grid --out results/2-beat-activation
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from . import data, evaluate
from .grid import activation, grid_scores
from .report import INK, INK2, GRID, SERIES, SURFACE, dataset_stats, new_fig, style

RUNS = {
    "1: drum detector": (["runs/nocomb", "runs/nocomb_s1", "runs/nocomb_s2"], "hit"),
    "2: beat activation": (["runs/grid", "runs/grid_s1", "runs/grid_s2"], "beat"),
}
COLOR = {"1: drum detector": SERIES["kick"], "2: beat activation": SERIES["snare"],
         "raw flux": SERIES["hihat"]}

MERMAID = """```mermaid
flowchart LR
  F["12 features<br/>93.75 blocks/s"] --> W["16-block window<br/>171 ms"]
  W --> M
  subgraph M["stage 1: MLP 192-128-64"]
    B["beat head<br/>is a beat in this block"]
    O["beat offset head<br/>sub-block position"]
    AUX["drum heads<br/>auxiliary, weight 0.3"]
  end
  M --> ACT["beat activation"] --> AC["stage 2: autocorrelation<br/>unchanged"] --> G["BPM + anchor"]
  LAB["labels: .alc MIDI grid<br/>beat 0 = clip start, exact"] -.-> B
```
"""


def collect(te):
    out = {}
    for name, (runs, key) in RUNS.items():
        rows, acts = [], []
        for r in runs:
            if not Path(r, "model.keras").exists():
                continue
            m = keras.models.load_model(Path(r) / "model.keras", compile=False)
            nz = np.load(Path(r) / "norm.npz")
            a = activation(m, te, nz["mean"], nz["scale"], key=key)
            rows.append(grid_scores(a, te))
            acts.append(a)
        out[name] = (rows, acts[0])
    flux = np.maximum(te.X[:, 5], te.X[:, 9])
    out["raw flux"] = ([grid_scores(flux, te)], flux)
    return out


def fig_compare(res, path):
    fig, ax = new_fig(5.4, 3.2)
    names = list(res)
    w = 0.38
    for k, (metric, label) in enumerate((("exact", "within 4%"), ("octave", "+octave"))):
        for i, n in enumerate(names):
            vals = [r[metric] for r in res[n][0]]
            ax.bar(i + (k - 0.5) * w, np.mean(vals) * 100, width=w * 0.92,
                   color=COLOR[n], alpha=1.0 if k == 0 else 0.45,
                   edgecolor=SURFACE, linewidth=1.5,
                   label=label if i == 0 else None)
            if len(vals) > 1:
                ax.errorbar(i + (k - 0.5) * w, np.mean(vals) * 100,
                            yerr=np.std(vals) * 100, color=INK2, linewidth=1.2,
                            capsize=3)
            ax.text(i + (k - 0.5) * w, np.mean(vals) * 100 + 2.2,
                    f"{np.mean(vals)*100:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(0, 100)
    style(ax, "", "% of takes", "Tempo, rendered clips (exact grid)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt_close(fig)


def fig_phase(res, te, path):
    fig, ax = new_fig(5.4, 3.2)
    for n, (_, act) in res.items():
        errs = []
        for t in np.unique(te.take):
            m = te.take == t
            idx = np.flatnonzero(te.beat[m][:, 0] > 0)
            if len(idx) < 8:
                continue
            bt = (idx + te.beat_off[m][idx, 0]) * evaluate.BLOCK_MS
            true_bpm = 60000.0 / float(np.median(np.diff(bt)))
            from . import tempo
            est = tempo.estimate(act[m])
            if tempo.tempo_ok(est.bpm, true_bpm)[1]:
                errs.append(tempo.grid_error_ms(est, true_bpm,
                                                float(bt[0] % (60000.0 / true_bpm))))
        a = np.sort(np.asarray(errs))
        y = np.arange(1, a.size + 1) / a.size
        ax.plot(a, y, color=COLOR[n], linewidth=2.0,
                label=f"{n}: median {np.median(a):.1f} ms, n={a.size}")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.axvline(120, color=GRID, linewidth=1.0, linestyle="--")
    ax.text(124, 0.94, "quarter beat (typical)", fontsize=7.5, color=INK2, va="top")
    style(ax, "grid phase error (ms)", "fraction of takes",
          "Phase, where tempo is right up to an octave")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt_close(fig)


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/2-beat-activation"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    te = data.load(a.features, "test")
    res = collect(te)
    fig_compare(res, a.out / "tempo.png")
    fig_phase(res, te, a.out / "phase.png")

    stats = dataset_stats(a.features, a.manifest)
    meta, sp = stats["meta"], stats["splits"]
    run = json.loads(Path("runs/grid/result.json").read_text())
    mb = lambda b: f"{b/1024/1024:.0f} MB"

    L = ["# 2: beat activation\n", MERMAID, "\n## Configuration\n", "| | |", "|---|---|",
         f"| front end | `FE_SPEC_VERSION {meta['fe_spec_version']}`, variant `{meta.get('fe_variant_name')}` |",
         f"| window | {run['window']} blocks ({run['window']*evaluate.BLOCK_MS:.0f} ms), lookahead {run['future']} |",
         f"| model | {run['hidden']} hidden, beat + offset heads, drum heads auxiliary ({run['aux']}) |",
         f"| size | {run['macs']:,} MAC/inference, ~{run['macs']/1024:.0f} KB int8 |",
         f"| labels | .alc MIDI grid, beat 0 = clip start |",
         f"| provenance | manifest `{meta['manifest_sha256'][:12]}`, IR `{(meta.get('ir_sha256') or '-')[:12]}` |\n",
         "## Dataset\n",
         "| split | takes | blocks | hours | size | beats |", "|---|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        if s not in sp:
            continue
        d = np.load(a.features / f"{s}.npz")
        L.append(f"| {s} | {sp[s]['takes']:,} | {sp[s]['blocks']:,} | {sp[s]['hours']:.2f} "
                 f"| {mb(sp[s]['bytes'])} | {int(d['beat'].sum()):,} |")
    L += [f"| **total** | {sum(v['takes'] for v in sp.values()):,} | "
          f"{sum(v['blocks'] for v in sp.values()):,} | {sum(v['hours'] for v in sp.values()):.2f} "
          f"| {mb(stats['features_bytes'])} | |\n",
          "## Grid: rendered clips, exact grid (identical stage 2, 3 seeds)\n",

          "| approach | tempo within 4% | +octave | bpm err | phase median | MACs |",
          "|---|---|---|---|---|---|"]
    for n, (rows, _) in res.items():
        ex = np.array([r["exact"] for r in rows]) * 100
        oc = np.array([r["octave"] for r in rows]) * 100
        ph = np.array([r["phase_median"] for r in rows])
        be = np.array([r["bpm_err"] for r in rows])
        sd = f" ±{ex.std():.1f}" if len(ex) > 1 else ""
        sdo = f" ±{oc.std():.1f}" if len(oc) > 1 else ""
        macs = run["macs"] if n.startswith("2") else (33152 if n.startswith("1") else 0)
        L.append(f"| {n} | {ex.mean():.1f}%{sd} | {oc.mean():.1f}%{sdo} | {be.mean():.2f}% "
                 f"| {ph.mean():.2f} ms | {macs:,} |")
    L += ["", "![tempo](tempo.png)", "", "![phase](phase.png)", ""]
    (a.out / "README.md").write_text("\n".join(L))
    print(f"-> {a.out}")
    for p in sorted(a.out.iterdir()):
        print(f"   {p.name:14} {p.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
