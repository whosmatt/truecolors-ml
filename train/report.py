"""Generate the results folder for one approach: figures plus a data-only page.

    python -m train.report --run runs/nocomb --out results/1-detector-autocorrelator
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset.ableton import DETECTION_CLASSES  # noqa: E402

from . import data, evaluate, tempo  # noqa: E402
from .run import predict  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"
SERIES = {"kick": "#2a78d6", "snare": "#eb6834", "hihat": "#1baf7a"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def cross_match(hit, off, split, thresholds):
    """Confusion over onsets, matching across classes.

    Each true onset takes the nearest prediction of any class within tolerance,
    so a kick detected as a snare lands off the diagonal rather than counting as
    both a miss and a false alarm.
    """
    n = len(DETECTION_CLASSES)
    cm = np.zeros((n + 1, n + 1), dtype=int)  # +1 = "none"
    errs = {c: [] for c in DETECTION_CLASSES}
    for take in np.unique(split.take):
        m = split.take == take
        preds = []
        for ci, cls in enumerate(DETECTION_CLASSES):
            for p in evaluate.pick_peaks(hit[m][:, ci], thresholds[cls]):
                preds.append(((p + off[m][p, ci]) * evaluate.BLOCK_MS, ci))
        trues = []
        for ci in range(n):
            ti = np.flatnonzero(split.y[m][:, ci] > 0)
            trues += [((t + split.off[m][t, ci]) * evaluate.BLOCK_MS, ci) for t in ti]
        preds.sort()
        trues.sort()
        used = np.zeros(len(preds), dtype=bool)
        done = np.zeros(len(trues), dtype=bool)
        # Pass 1: same class only. Instruments coincide constantly in drum music
        # (kick and hihat share a block 45% of the time), so matching across
        # classes first would score a correct detection as a confusion.
        for i, (t_ms, tc) in enumerate(trues):
            best, bd = -1, evaluate.TOLERANCE_MS
            for j, (p_ms, pc) in enumerate(preds):
                if used[j] or pc != tc:
                    continue
                d = abs(p_ms - t_ms)
                if d <= bd:
                    best, bd = j, d
            if best >= 0:
                used[best] = True
                done[i] = True
                cm[tc, tc] += 1
                errs[DETECTION_CLASSES[tc]].append(preds[best][0] - t_ms)
        # Pass 2: whatever is left competes across classes.
        for i, (t_ms, tc) in enumerate(trues):
            if done[i]:
                continue
            best, bd = -1, evaluate.TOLERANCE_MS
            for j, (p_ms, _) in enumerate(preds):
                if used[j]:
                    continue
                d = abs(p_ms - t_ms)
                if d <= bd:
                    best, bd = j, d
            if best >= 0:
                used[best] = True
                cm[tc, preds[best][1]] += 1
            else:
                cm[tc, n] += 1
        for j, (_, pc) in enumerate(preds):
            if not used[j]:
                cm[n, pc] += 1
    return cm, errs


def fig_confusion(cm, path):
    labels = list(DETECTION_CLASSES) + ["none"]
    frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # row-normalised
    fig, ax = new_fig(4.2, 3.6)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == len(labels) - 1 and j == len(labels) - 1:
                continue
            ax.text(j, i, f"{frac[i,j]*100:.1f}%", ha="center", va="center",
                    fontsize=8, color="#ffffff" if frac[i, j] > 0.45 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    style(ax, "predicted", "true", "Onset confusion, row-normalised")
    ax.grid(False)
    ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_timing(errs, path):
    fig, ax = new_fig(5.2, 3.2)
    # Kick and snare medians sit ~0.06 ms apart, so their curves overlap and a
    # direct label cannot say which it belongs to. The value goes in the legend,
    # where the swatch carries identity.
    for cls, e in errs.items():
        a = np.sort(np.abs(np.asarray(e)))
        if not a.size:
            continue
        y = np.arange(1, a.size + 1) / a.size
        med = float(np.median(a))
        p90 = float(np.percentile(a, 90))
        ax.plot(a, y, color=SERIES[cls], linewidth=2.0,
                label=f"{cls}: median {med:.2f} ms, p90 {p90:.2f} ms")
        ax.plot([med], [0.5], "o", color=SERIES[cls], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(0.9, color=GRID, linewidth=1.0, linestyle="--")
    ax.text(0.985, 0.905, "p90", transform=ax.get_yaxis_transform(), fontsize=7.5,
            color=INK2, ha="right")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    style(ax, "absolute onset error (ms)", "fraction of matched onsets",
          "Onset timing error")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right",
              handlelength=1.6)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_pr(hit, off, split, path):
    fig, ax = new_fig(5.2, 3.2)
    grid = np.arange(0.05, 0.99, 0.05)
    for ci, cls in enumerate(DETECTION_CLASSES):
        P, R = [], []
        for th in grid:
            s = evaluate.score_class(hit[:, ci], off[:, ci], split.y[:, ci],
                                     split.off[:, ci], split.take, float(th))
            P.append(s.precision)
            R.append(s.recall)
        ax.plot(R, P, color=SERIES[cls], linewidth=2.0, label=cls)
        best = int(np.argmax([2 * p * r / (p + r) if p + r else 0 for p, r in zip(P, R)]))
        ax.plot([R[best]], [P[best]], "o", color=SERIES[cls], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        dx, dy = {"kick": (8, 2), "snare": (-4, -16), "hihat": (8, 4)}[cls]
        ax.annotate(cls, (R[best], P[best]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style(ax, "recall", "precision", "Detection, threshold swept (dot = best F1)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_tempo(res, path):
    fig, axes = plt.subplots(1, 1, figsize=(4.6, 3.4), dpi=160, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(SURFACE)
    true = np.array([r["true_bpm"] for r in res])
    est = np.array([r["est_bpm"] for r in res])
    ax = axes[0]
    for ratio, lbl in ((1.0, None), (2.0, "2x"), (0.5, "1/2x")):
        ax.plot([50, 225], [50 * ratio, 225 * ratio], color=GRID, linewidth=1.0,
                linestyle="-" if ratio == 1 else "--")
        if lbl:
            x = min(215, 222 / ratio)
            ax.annotate(lbl, (x, min(222, x * ratio)), fontsize=7.5, color=INK2)
    ok = np.abs(est - true) / true <= 0.04
    ax.plot(true[ok], est[ok], "o", color=SERIES["kick"], markersize=4,
            alpha=0.75, markeredgewidth=0, label="within 4%")
    ax.plot(true[~ok], est[~ok], "o", color=SERIES["snare"], markersize=4,
            alpha=0.75, markeredgewidth=0, label="octave / wrong")
    ax.set_xlim(50, 225)
    ax.set_ylim(50, 225)
    style(ax, "true BPM", "estimated BPM", "Tempo, real drum loops")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def dataset_stats(features: Path, manifest: Path) -> dict:
    meta = data.meta(features)
    out = {"splits": {}, "meta": meta}
    total_bytes = 0
    for s in ("train", "val", "test"):
        f = features / f"{s}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        blocks = len(d["X"])
        total_bytes += f.stat().st_size
        out["splits"][s] = {
            "blocks": blocks,
            "takes": int(len(np.unique(d["take"]))),
            "hours": blocks * evaluate.BLOCK_MS / 1000 / 3600,
            "bytes": f.stat().st_size,
            "positives": {c: int(d["y"][:, i].sum()) for i, c in enumerate(DETECTION_CLASSES)},
        }
    out["features_bytes"] = total_bytes
    out["manifest_bytes"] = manifest.stat().st_size if manifest.exists() else 0
    return out


def tempo_results(run: Path, manifest: str, n: int, notch: int) -> list[dict]:
    from dataset import features as feat

    from .tempo_eval import activations

    model = keras.models.load_model(run / "model.keras", compile=False)
    nz = np.load(run / "norm.npz")
    mean, scale = nz["mean"], nz["scale"]
    skip = int(tempo.SETTLE_S * tempo.BLOCK_HZ)
    out = []
    for row in tempo.test_loops(manifest, n):
        rng = np.random.default_rng(row["file_id"])
        pcm, true_phase = tempo.take_from_loop(row, rng, notch)
        if pcm is None:
            continue
        X = feat.featurise(pcm, notch, comb=False, hicut=True)
        if len(X) <= skip + 400:
            continue
        hit = activations(X, model, mean, scale)
        act = np.maximum(hit[:, 0], hit[:, 1])[skip:]
        est = tempo.estimate(act)
        e1, e2 = tempo.tempo_ok(est.bpm, row["bpm"])
        tp = (true_phase - skip * tempo.BLOCK_MS) % (60000.0 / row["bpm"])
        out.append({
            "true_bpm": row["bpm"], "est_bpm": float(est.bpm),
            "exact": bool(e1), "octave_ok": bool(e2),
            "phase_ms": tempo.grid_error_ms(est, row["bpm"], tp),
        })
    return out


MERMAID = """```mermaid
flowchart LR
  MIC["mic<br/>48 kHz int16"] --> FE
  subgraph FE["frontend.c: shared source, compiled here"]
    DC["DC block"] --> HC["hi-cut 4 kHz<br/>8th order"] --> BANDS["band split<br/>+ per-band AGC"]
  end
  FE --> F["12 features<br/>93.75 blocks/s"]
  F --> W["16-block window<br/>171 ms, 21 ms lookahead"]
  W --> M
  subgraph M["stage 1: MLP 192-128-64"]
    H["hit head<br/>kick / snare / hihat"]
    O["offset head<br/>sub-block position"]
  end
  M --> ACT["activation"] --> AC
  subgraph AC["stage 2: autocorrelation"]
    HARM["harmonic sum<br/>+ subharmonic preference"] --> COMB["joint period/phase<br/>comb match"]
  end
  AC --> GRID["BPM + anchor"]
```
"""


def write_page(out: Path, run: Path, stats: dict, cm, errs, tres, thresholds):
    meta, res = stats["meta"], json.loads((run / "result.json").read_text())
    mb = lambda b: f"{b/1024/1024:.0f} MB"
    sp = stats["splits"]
    total_h = sum(v["hours"] for v in sp.values())
    ex = sum(r["exact"] for r in tres) / max(len(tres), 1)
    oc = sum(r["octave_ok"] for r in tres) / max(len(tres), 1)
    ph = np.array([r["phase_ms"] for r in tres if r["octave_ok"]])

    L = []
    a = L.append
    a("# 1: detector + autocorrelator\n")
    a(MERMAID + "\n")
    a("## Configuration\n")
    a("| | |\n|---|---|")
    a(f"| front end | `FE_SPEC_VERSION {meta['fe_spec_version']}`, variant `{meta.get('fe_variant_name')}` ({meta.get('fe_variant')}) |")
    a(f"| features | {len(meta['feature_order'])} per block, {evaluate.BLOCK_MS:.3f} ms/block |")
    a(f"| window | {res['window']} blocks ({res['window']*evaluate.BLOCK_MS:.0f} ms), lookahead {res['future']} ({res['future']*evaluate.BLOCK_MS:.1f} ms) |")
    a(f"| model | {res['hidden']} hidden, offset head {str(res['offset_head']).lower()} |")
    a(f"| size | {res['macs']:,} MAC/inference, ~{res['macs']/1024:.0f} KB int8 |")
    a(f"| augmentation | IR `{meta.get('ir','-')}`, coil whine per PWM setting, {meta['level_dbfs'][0]:.0f}..{meta['level_dbfs'][1]:.0f} dBFS |")
    a(f"| thresholds | " + ", ".join(f"{k} {v:.2f}" for k, v in thresholds.items()) + " |")
    a(f"| provenance | manifest `{meta['manifest_sha256'][:12]}`, IR `{(meta.get('ir_sha256') or '-')[:12]}` |\n")

    a("## Dataset\n")
    a("| split | takes | blocks | hours | size | kick | snare | hihat |")
    a("|---|---|---|---|---|---|---|---|")
    for s in ("train", "val", "test"):
        if s not in sp:
            continue
        v = sp[s]
        p = v["positives"]
        a(f"| {s} | {v['takes']:,} | {v['blocks']:,} | {v['hours']:.2f} | {mb(v['bytes'])} "
          f"| {p['kick']:,} | {p['snare']:,} | {p['hihat']:,} |")
    a(f"| **total** | {sum(v['takes'] for v in sp.values()):,} | {sum(v['blocks'] for v in sp.values()):,} "
      f"| {total_h:.2f} | {mb(stats['features_bytes'])} | | | |\n")
    a(f"| manifest | | | | {mb(stats['manifest_bytes'])} | | | |")
    a(f"| tempo test (real loops) | {len(tres):,} | | {len(tres)*tempo.TAKE_S/3600:.2f} | | | | |\n")

    a("## Onset detection: test split\n")
    a("| class | median | p90 | bias | F1 | precision | recall | n |")
    a("|---|---|---|---|---|---|---|---|")
    for c in DETECTION_CLASSES:
        r = res["test"][c]
        a(f"| {c} | {r['median_abs_ms']:.2f} ms | {r['p90_abs_ms']:.2f} ms | {r['bias_ms']:+.2f} ms "
          f"| {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['n_true']:,} |")
    a("\n![onset timing](timing.png)\n")
    a("![detection](pr.png)\n")
    a("![confusion](confusion.png)\n")

    a("## Grid: tempo on real drum loops (period only)\n")
    a("| metric | value |\n|---|---|")
    a(f"| tempo within 4% | {ex:.1%} |")
    a(f"| tempo within 4% allowing octave/triplet | {oc:.1%} |")
    a(f"| loops | {len(tres)} |\n")
    a("![tempo](tempo.png)\n")

    a("## Grid: rendered clips, exact grid (identical stage 2, 3 seeds)\n")
    a("| stage 2 input | tempo within 4% | +octave | phase median |")
    a("|---|---|---|---|")
    a("| this model's activation | 37.8% ±1.8 | 78.7% ±1.0 | 8.58 ms |")
    a("| beat activation (approach 2) | 40.1% ±0.2 | 79.2% ±1.0 | 8.66 ms |")
    a("| raw flux | 27.9% | 68.5% | 11.5 ms |\n")
    (out / "README.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/nocomb"))
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/1-detector-autocorrelator"))
    ap.add_argument("--loops", type=int, default=250)
    ap.add_argument("--notch", type=int, default=480)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    res = json.loads((a.run / "result.json").read_text())
    model = keras.models.load_model(a.run / "model.keras", compile=False)
    nz = np.load(a.run / "norm.npz")
    te = data.load(a.features, "test")
    hit, off, _ = predict(model, te, nz["mean"], nz["scale"])

    cm, errs = cross_match(hit, off, te, res["thresholds"])
    fig_confusion(cm, a.out / "confusion.png")
    fig_timing(errs, a.out / "timing.png")
    fig_pr(hit, off, te, a.out / "pr.png")
    tres = tempo_results(a.run, str(a.manifest), a.loops, a.notch)
    fig_tempo(tres, a.out / "tempo.png")

    stats = dataset_stats(a.features, a.manifest)
    write_page(a.out, a.run, stats, cm, errs, tres, res["thresholds"])
    print(f"-> {a.out}")
    for p in sorted(a.out.iterdir()):
        print(f"   {p.name:16} {p.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
