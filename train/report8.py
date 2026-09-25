"""Results page for the deploy candidate (music v3): the full chain, the training
pool, the music-positive test behind it, and float vs int8 on every test set.

    python -m train.report8 [--recompute]
"""

import argparse
import json
from pathlib import Path

from . import gpu  # noqa: F401  must precede keras

import keras
import matplotlib.pyplot as plt
import numpy as np

from . import data, export, gridmetrics, mres, musiceval, scoreboard, seq, tempo
from .compare import tempo_score
from .report import GRID, INK, INK2, SERIES, new_fig, style

OUT = Path("results/8-deploy-candidate")
BLOCK_S = 512 / 48000
EXPORT = Path("export/music4")
RUN = Path("runs/m_cand_s0")  # seed 0 by rule, not the better test seed
V2 = Path("runs/g_music2_s0")
V2_TFLITE = Path("export/music3/model.tflite")

# music-positive test: (label, run prefix, flags)
MUSIC_TEST = [
    ("music v2, delivered", "g_music2", "drum loops as music, grid masked; music weight 0.3"),
    ("v3: drum grid + melodic phase-free (chosen)", "m_cand", "music weight 0.3"),
    ("v3 + melodic loops as music", "n_melmus", "music weight 0.3"),
    ("v3 + melodic loops as music", "n_melmus_p15", "music weight 0.15"),
    ("v3 + melodic loops as music", "n_melmus_p08", "music weight 0.08"),
]
GROUPS = ["real loops", "melodic", "rendered", "non-music", "silence"]
GATE_T = 0.7
TEST_TS = (0.7, 0.8)

POOL = [
    # corpus, path, beat/offset, drum, music, melodic loss
    ("rendered .alc clips, 2026-09-24 index", "data/features4", "exact grid", "per hit", "1, silence 0", "-"),
    ("real drum loops, file-start grid", "data/loops_grid", "file-start grid", "masked", "1", "-"),
    ("melodic loops", "data/melodic", "masked", "masked", "1", "phase-free, known tempo"),
    ("non-music takes", "data/noise", "masked", "masked", "0", "-"),
]

MERMAID = """```mermaid
flowchart TB
  MIC["mic, I2S 48 kHz int16<br/>512-sample blocks, 93.75/s"] --> FE
  subgraph FE["frontend.c, FE_SPEC_VERSION 2, variant hicut"]
    direction LR
    DC["DC block"] --> HC["hi-cut 4 kHz<br/>8th order"] --> BS["band split"] --> AGC["per-band AGC"]
  end
  FE --> F["12 features per block<br/>level, bands x3, rms,<br/>kick flux 40-80 / 80-140 / 140-220 Hz,<br/>fund_rms, mid flux 0.2-2 kHz,<br/>treble flux 2-4 kHz, spl_db"]
  F --> RING["ring buffer<br/>272 blocks x 12"]
  RING --> FINE["fine: 16 frames<br/>t-13 .. t+2, single blocks"]
  RING --> MID["mid: 16 frames<br/>4-block means, t-17 .. t-77"]
  RING --> COARSE["coarse: 12 frames<br/>16-block means, t-77 .. t-269"]
  FINE & MID & COARSE --> NORM["528 inputs<br/>(x - mean) / scale, int8"]
  NORM --> D0["Dense 528 -> 128, ReLU"] --> D1["Dense 128 -> 64, ReLU"]
  D1 --> BEAT["beat, sigmoid"]
  D1 --> OFF["beat_offset, sigmoid<br/>position in block"]
  D1 --> HIT["hit x4, sigmoid<br/>kick / snare / hihat / none"]
  D1 --> MUS["music, sigmoid"]
  BEAT & OFF --> AC["stage 2: autocorrelation,<br/>harmonic sum, comb phase fit"]
  AC --> G["BPM + beat anchor"]
  MUS --> GATE["gate: smoothed over seconds,<br/>threshold 0.7"]
```"""


def pool_rows():
    out = []
    for name, path, beat, drum, music, mel in POOL:
        sp = data.meta(Path(path))["splits"]["train"]
        out.append(f"| {name} | {sp['takes']:,} | {sp['blocks'] * BLOCK_S / 3600:.1f} | "
                   f"{beat} | {drum} | {music} | {mel} |")
    return out


def gate(runs, tflites=None, split="val", ts=(GATE_T,)):
    """-> {threshold: {group: [% correct side per run]}}."""
    spec = mres.Spec()
    corp = {"rendered": data.load(Path("data/features4"), split),
            "real loops": data.load(Path("data/loops4"), split),
            "melodic": data.load(Path("data/melodic"), split),
            "noise": data.load(Path("data/noise"), split)}
    out = {str(t): {g: [] for g in GROUPS} for t in ts}
    for k, run in enumerate(runs):
        nz = np.load(Path(run) / "norm.npz")
        m = keras.models.load_model(Path(run) / "model.keras", compile=False)
        if tflites:
            m = musiceval.Int8(tflites[k], export_heads(Path(run)))
        rng = np.random.default_rng(0)
        for n, s in corp.items():
            med, lvl = musiceval.take_medians(m, s, nz["mean"], nz["scale"], spec, rng)
            for t in ts:
                o = out[str(t)]
                if n == "noise":
                    o["non-music"].append(float(np.mean(med[lvl >= musiceval.SILENT_DB] < t) * 100))
                    o["silence"].append(float(np.mean(med[lvl < musiceval.SILENT_DB] < t) * 100))
                else:
                    o[n].append(float(np.mean(med >= t) * 100))
    return out


def export_heads(run):
    m = keras.models.load_model(run / "model.keras", compile=False)
    return scoreboard_heads(m)


def scoreboard_heads(m):
    return list(m.output_names) if hasattr(m, "output_names") else list(m.output.keys())


def int8_scores(run, blob):
    """Float and int8 on every test set, 461-block window as train.compare."""
    spec = mres.Spec()
    m = keras.models.load_model(run / "model.keras", compile=False)
    nz = np.load(run / "norm.npz")
    heads = scoreboard_heads(m)
    sets = {"clips": ("data/features4", True), "loops_grid": ("data/loops_grid", True),
            "melodic_grid": ("data/melodic_grid", True), "loops": ("data/loops", False),
            "melodic": ("data/melodic", False)}
    out = {}
    from .compare import pos_in_take
    for name, (path, grid) in sets.items():
        s = data.load_many([Path(path)], "test")
        keep = pos_in_take(s) >= 461
        af = scoreboard.activation(m, s, nz["mean"], nz["scale"], spec)[0]
        aq = export.run_tflite(blob, s, nz["mean"], nz["scale"], spec, heads)["beat"]
        r = {}
        for lbl, a in (("float32", af), ("int8", aq)):
            a = np.where(keep, a, 0.0).astype(np.float32)
            r[lbl] = gridmetrics.per_take(a, s) if grid else tempo_score(a, s)
        c = mres.valid_centres(s.take, spec)
        r["correlation"] = float(np.corrcoef(af[c], aq[c])[0, 1])
        out[name] = r
        print(name, {k: (v if k == "correlation" else {x: round(y, 3) for x, y in v.items()})
                     for k, v in r.items()}, flush=True)
    return out


def fmt(v, unit="%", nd=1):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return "-"
    return f"{v.mean():.{nd}f}{unit}" + (f" ±{v.std():.{nd}f}" if v.size > 1 else "")


def plot_gate(g2, g3, path):
    fig, ax = new_fig(6.4, 3.0)
    x = np.arange(len(GROUPS))
    w = 0.36
    for k, (lbl, g, col) in enumerate((("music v2 (deployed)", g2, SERIES["kick"]),
                                       ("music v3 (this candidate)", g3, SERIES["snare"]))):
        v = [np.mean(g[n]) for n in GROUPS]
        ax.bar(x + (k - 0.5) * (w + 0.02), v, width=w, color=col, label=lbl, zorder=2)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * (w + 0.02), vi + 1.5, f"{vi:.0f}", ha="center",
                    fontsize=7, color=INK2)
    labels = [f"{n}\n{'kept' if n in ('real loops', 'melodic', 'rendered') else 'rejected'}"
              for n in GROUPS]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    style(ax, ylabel=f"% of takes, int8, threshold {GATE_T}")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sets(s2, s3, path):
    """Usable grid (rendered, drum, melodic) and tempo, int8, v2 against v3."""
    items = [("rendered clips\nusable", "clips", "usable"),
             ("real drum loops\nusable", "loops_grid", "usable"),
             ("melodic loops\nusable", "melodic_grid", "usable"),
             ("real drum loops\ntempo", "loops", "tempo"),
             ("melodic loops\ntempo", "melodic", "tempo")]
    fig, ax = new_fig(6.4, 3.0)
    x = np.arange(len(items))
    w = 0.36
    for k, (lbl, sc, col) in enumerate((("music v2 (deployed)", s2, SERIES["kick"]),
                                        ("music v3 (this candidate)", s3, SERIES["snare"]))):
        v = [sc[t]["int8"][f] * 100 for _, t, f in items]
        ax.bar(x + (k - 0.5) * (w + 0.02), v, width=w, color=col, label=lbl, zorder=2)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * (w + 0.02), vi + 1.2, f"{vi:.0f}", ha="center",
                    fontsize=7, color=INK2)
    ax.set_xticks(x, [i[0] for i in items], fontsize=7)
    ax.set_ylim(0, 80)
    style(ax, ylabel="% of test takes, int8")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trace(run, blob, path, seconds=6.0):
    """One real drum and one melodic test loop: int8 beat activation, labelled
    beats and the stage-2 grid, after the 461-block settle."""
    spec = mres.Spec()
    m = keras.models.load_model(run / "model.keras", compile=False)
    nz = np.load(run / "norm.npz")
    heads = scoreboard_heads(m)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.8), dpi=160, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, path_) in zip(axes, (("real drum loop", "data/loops_grid"),
                                          ("melodic loop", "data/melodic_grid"))):
        s = data.load_many([Path(path_)], "test")
        a = export.run_tflite(blob, s, nz["mean"], nz["scale"], spec, heads)["beat"]
        rng = np.random.default_rng(3)
        for g in rng.permutation(seq.takes(s)):
            est = tempo.estimate(a[g])
            idx = np.flatnonzero(s.beat[g][:, 0] > 0)
            bt = (idx + s.beat_off[g][idx, 0]) * 1000 * BLOCK_S
            true_bpm = 60000.0 / float(np.median(np.diff(bt)))
            ok = tempo.tempo_ok(est.bpm, true_bpm)[0] and \
                tempo.grid_error_ms(est, true_bpm, float(bt[0] % (60000 / true_bpm))) <= 25
            if ok:
                break
        t0 = 461
        n = int(seconds / BLOCK_S)
        tt = np.arange(n) * BLOCK_S
        ax.plot(tt, a[g][t0:t0 + n], color=SERIES["kick"], linewidth=1.0, zorder=3,
                label="beat activation, int8")
        for b in bt / 1000:
            if t0 * BLOCK_S <= b < (t0 + n) * BLOCK_S:
                ax.axvline(b - t0 * BLOCK_S, color=INK2, linewidth=0.8,
                           linestyle=(0, (3, 3)), zorder=1)
        grid = est.phase_ms / 1000 + np.arange(200) * est.lag * BLOCK_S
        grid = grid[(grid >= t0 * BLOCK_S) & (grid < (t0 + n) * BLOCK_S)] - t0 * BLOCK_S
        ax.scatter(grid, np.full(len(grid), 1.1), marker="v", s=14, color=SERIES["snare"],
                   zorder=4, label="stage-2 grid")
        ax.set_ylim(0, 1.2)
        style(ax, ylabel="activation",
              title=f"{title}: labelled {true_bpm:.1f} BPM, stage 2 {est.bpm:.1f} BPM")
    axes[-1].set_xlabel("seconds", color=INK2, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK2, linewidth=0.8, linestyle=(0, (3, 3))))
    l.append("labelled beats")
    fig.legend(h, l, frameon=False, fontsize=7, labelcolor=INK, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "final.json"
    meta = json.loads((EXPORT / "model_meta.json").read_text())
    blob = (EXPORT / "model.tflite").read_bytes()
    blob2 = V2_TFLITE.read_bytes()
    if cache.exists() and not a.recompute:
        fin = json.loads(cache.read_text())
    else:
        test = {}
        for _, prefix, _ in MUSIC_TEST:
            runs = sorted(str(p) for p in Path("runs").glob(f"{prefix}_s[0-9]"))
            test[prefix] = {"gate": gate(runs, ts=TEST_TS), "runs": runs}
        fin = {"music_test": test,
               "gate_int8": {"v2": gate([V2], [V2_TFLITE]), "v3": gate([RUN], [EXPORT / "model.tflite"])},
               "sets": {"v2": int8_scores(V2, blob2), "v3": int8_scores(RUN, blob)}}
        cache.write_text(json.dumps(fin, indent=1))
    grid = json.loads(Path("results/7-loop-grids/candidate.json").read_text())

    run = json.loads((RUN / "result.json").read_text())
    L = ["# 8: deploy candidate, music v3", "", MERMAID, "",
         "## Configuration", "", "| | |", "|---|---|",
         f"| front end | `FE_SPEC_VERSION` 2 (recorded {meta['fe_spec_version']}), variant `{meta['fe_variant_name']}` ({meta['fe_variant']}) |",
         f"| context | {meta['context']['seconds']:.2f} s, {meta['context']['frames']} frames, {meta['context']['inputs']} inputs, ring {meta['context']['ring_blocks']} blocks, lookahead {meta['context']['lookahead_blocks']} blocks ({meta['context']['lookahead_blocks'] * BLOCK_S * 1000:.1f} ms) |",
         "| model | MLP 528-128-64, ReLU, dropout 0.1 in training; heads " + ", ".join(
             f"{k} ({v})" for k, v in meta["outputs"].items()) + " |",
         f"| size | {meta['macs']:,} MACs per block, {meta['tflite_bytes']:,} B int8 TFLite |",
         f"| run | `{RUN}`, seed {run['seed']}, {run['minutes']:.1f} min on RTX 3070, {run['train_blocks']:,} training blocks |",
         "| losses | beat: BCE, positive weight 20; offset: MSE on beats, weight 5; drum: BCE, weight 0.3; "
         f"music: BCE, positive weight {run['music_pos']}; melodic: phase-free BCE, weight {run['mil'][0]} |",
         f"| export | int8 in and out, calibration on rendered + noise + drum loops + melodic loops; sha256 `{meta['sha256'][:12]}` |",
         f"| provenance | manifest `{meta['trained_from']['manifest_sha256'][:12]}`, IR `{(meta['trained_from']['ir_sha256'] or '')[:12]}`, commit `{meta['trained_from']['repo_commit'][:7]}` |",
         "", "## Training pool", "",
         "| corpus | train takes | hours | beat + offset | drum | music | melodic loss |",
         "|---|---|---|---|---|---|---|", *pool_rows(), "",
         "## Music positives: melodic loops as music, gate per-take median, val, float", "",
         "| configuration | music weight | threshold | " + " | ".join(
             f"{g} {'kept' if g in ('real loops', 'melodic', 'rendered') else 'rejected'}" for g in GROUPS)
         + " | rendered usable | drum usable | melodic tempo | seeds |",
         "|---|---|---|" + "---|" * (len(GROUPS) + 4)]
    for th in TEST_TS:
        for lbl, prefix, note in MUSIC_TEST:
            t = fin["music_test"][prefix]
            rows = [v for k, v in grid.items() if k.rsplit("/", 1)[-1].rsplit("_s", 1)[0] == prefix]
            gv = lambda test, f: [r[test][f] * 100 for r in rows]
            L.append(f"| {lbl} | {note.rsplit(' ', 1)[-1]} | {th} | "
                     + " | ".join(fmt(t["gate"][str(th)][g]) for g in GROUPS)
                     + f" | {fmt(gv('clips', 'usable'))} | {fmt(gv('loops_grid', 'usable'))} | "
                     f"{fmt(gv('melodic', 'tempo'))} | {len(t['runs'])} |")
    L += ["", "## Float against int8, test split, 461-block window", "",
          "| test set | metric | v2 float | v2 int8 | v3 float | v3 int8 | v3 activation correlation |",
          "|---|---|---|---|---|---|---|"]
    for name, metric in (("clips", "usable"), ("loops_grid", "usable"), ("melodic_grid", "usable"),
                         ("loops", "tempo"), ("melodic", "tempo")):
        v2, v3 = fin["sets"]["v2"][name], fin["sets"]["v3"][name]
        label = {"clips": "rendered clips, 2026-09-24", "loops_grid": "real drum loops",
                 "melodic_grid": "melodic loops", "loops": "real drum loops",
                 "melodic": "melodic loops"}[name]
        L.append(f"| {label} | {metric} | {v2['float32'][metric]*100:.1f}% | {v2['int8'][metric]*100:.1f}% | "
                 f"{v3['float32'][metric]*100:.1f}% | {v3['int8'][metric]*100:.1f}% | {v3['correlation']:.4f} |")
    for name, label in (("clips", "rendered clips"), ("loops_grid", "real drum loops"),
                        ("melodic_grid", "melodic loops")):
        v2, v3 = fin["sets"]["v2"][name], fin["sets"]["v3"][name]
        L.append(f"| {label} | phase, tempo exact | {v2['float32']['phase_exact']:.1f} ms | "
                 f"{v2['int8']['phase_exact']:.1f} ms | {v3['float32']['phase_exact']:.1f} ms | "
                 f"{v3['int8']['phase_exact']:.1f} ms | |")
    L += ["", "![sets](sets.png)", "",
          f"## Music gate, int8, val, threshold {GATE_T}", "",
          "| model | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    for k, lbl in (("v2", "music v2 (deployed)"), ("v3", "music v3")):
        L.append(f"| {lbl} | " + " | ".join(f"{fin['gate_int8'][k][str(GATE_T)][g][0]:.1f}%"
                                              for g in GROUPS) + " |")
    L += ["", "![gate](gate.png)", "", "## Example: int8 beat activation on held-out loops, first usable take of a seeded shuffle", "",
          "![trace](trace.png)", ""]
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    plot_gate(fin["gate_int8"]["v2"][str(GATE_T)], fin["gate_int8"]["v3"][str(GATE_T)], OUT / "gate.png")
    plot_sets(fin["sets"]["v2"], fin["sets"]["v3"], OUT / "sets.png")
    plot_trace(RUN, blob, OUT / "trace.png")
    print(f"-> {OUT}/README.md")


if __name__ == "__main__":
    main()
