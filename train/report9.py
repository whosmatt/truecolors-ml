"""Results pages 9 (mel and real songs, model d0) and 10 (two-stage model g),
from the stored evaluation files.

    python -m train.report9
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .report import INK, INK2, SERIES, new_fig, style

R = Path("runs")
OUT9 = Path("results/9-mel-and-songs")
OUT10 = Path("results/10-two-stage")
EXPORT = Path("export/twostage1")
SETS = [("rendered usable", "clips", "usable"), ("rendered octave", "clips", "octave"),
        ("drum tempo", "loops", "tempo"), ("melodic tempo", "melodic", "tempo"),
        ("drum grid usable", "loops_grid", "usable"), ("melodic grid usable", "melodic_grid", "usable"),
        ("rendered phase, ms", "clips", "phase_exact")]

# Unlabelled-song lock share (dataset.songgrid re-analysis, share of 12 s sections the model
# locks), paired against the row's control over the 189 songs still unlabelled on 2026-09-28.
# Those songs are labelled now, so the set cannot be rebuilt from files: values as measured.
LOCK = {
    "a1_wide": "+1.2 [-1.1, +3.4]", "a2_ctx24": "-2.4 [-5.1, +0.4]", "a3_both": "+0.9 [-1.8, +3.6]",
    "a4_songs3": "+1.8 [-0.8, +4.2]", "b2_p12": "+4.5 [+2.1, +6.9]", "b1_flux16": "+1.8 [-0.6, +4.3]",
    "n_song": "+2.9 [+0.6, +5.1] vs v3",
}

BEATNET_PARAMS = [("Conv1d 1 -> 2, kernel 10", 22), ("Linear 262 -> 150", 39450),
                  ("LSTM 2 x 150", 362400), ("Linear 150 -> 3", 453)]


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def seeds(d, name, n=2):
    return [d[f"runs/{name}_s{s}"] for s in range(n) if f"runs/{name}_s{s}" in d]


def mstd(v, pct=True, nd=1):
    v = np.asarray([x for x in v if x is not None], float) * (100 if pct else 1)
    if not len(v):
        return "-"
    return f"{v.mean():.{nd}f}" + (f" ±{v.std():.{nd}f}" if len(v) > 1 else "")


def set_row(rows):
    return [mstd([r[t][f] for r in rows], pct=(f != "phase_exact")) for _, t, f in SETS]


def songs_row(se, name, splits=("test", "val")):
    rows = seeds(se, name)
    return [mstd([r[sp]["usable"] for r in rows]) for sp in splits]


def stream_row(st, name, corpus, v="hold8"):
    rows = [st.get(f"runs/{name}_s{s}:{corpus}:test:{v}") for s in (0, 1)]
    rows = [r for r in rows if r]
    if not rows:
        return ["-"] * 4
    return [mstd([r["usable"] for r in rows])] + ([
        mstd([r["relock_median_s"] for r in rows], pct=False) + " s",
        mstd([r["stale_s_first10"] for r in rows], pct=False) + " s",
        mstd([r["never"] for r in rows])] if "relock_median_s" in rows[0] else ["-"] * 3)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cv_pool(se, prefix, folds=5):
    rows = [se[f"runs/{prefix}_f{k}_s0"]["test"] for k in range(folds)]
    w = np.array([r["windows"] for r in rows], float)
    pool = lambda key: float(np.sum([r[key] for r in rows] * w) / w.sum())
    return {"songs": sum(r["songs"] for r in rows), "windows": int(w.sum()),
            "usable": pool("usable"), "tempo": pool("tempo"), "octave": pool("octave"),
            "folds": [r["usable"] for r in rows]}


def cv_stream(st, prefix, tag, corpus_fmt, folds=5):
    rows = [st.get(f"runs/{prefix}_f{k}_s0:{corpus_fmt.format(k=k)}:test:hold8") for k in range(folds)]
    rows = [r for r in rows if r]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}


def cv_sets(path):
    d = load(path)
    by = {}
    for k, v in d.items():
        by.setdefault(k.split("/")[1].split("_f")[0], []).append(v)
    return by


def bar(ax, groups, series, ylim, ylabel):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for k, (lbl, vals, col) in enumerate(series):
        pos = x + (k - (len(series) - 1) / 2) * w
        ax.bar(pos, vals, width=w * 0.92, color=col, label=lbl, zorder=2)
        for xi, vi in zip(pos, vals):
            ax.text(xi, vi + ylim * 0.012, f"{vi:.0f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, groups, fontsize=7)
    ax.set_ylim(0, ylim)
    style(ax, ylabel=ylabel)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(series))


COLORS = [SERIES["kick"], SERIES["snare"], SERIES["hihat"], "#8a6fd1", "#9a9892"]

MERMAID9 = """```mermaid
flowchart TB
  MIC["mic, I2S 48 kHz int16<br/>512-sample blocks, 93.75/s"] --> FE & MEL
  subgraph FE["frontend.c, FE_SPEC_VERSION 2, variant hicut"]
    direction LR
    DC["DC block"] --> HC["hi-cut 4 kHz"] --> BS["band split"] --> AGC["per-band AGC"]
  end
  subgraph MEL["melflux.c"]
    direction LR
    W["1024 Hann window<br/>ending at the block"] --> FFT["rfft, power"] --> FB["16 bands, 150 Hz - 4 kHz<br/>(40 HTK triangles merged)"] --> LOG["log10"] --> FL["positive flux"]
  end
  FE --> F["12 features per block"]
  MEL --> M["16 mel flux per block"]
  F & M --> X["28 per block, (x - mean) / scale"]
  X --> P["projection 28 -> 12, per block, linear"]
  P --> RING["ring 272 blocks x 12"]
  RING --> FINE["fine: 16 frames, t-13 .. t+2"]
  RING --> MID["mid: 16 x 4-block means"]
  RING --> COARSE["coarse: 12 x 16-block means"]
  FINE & MID & COARSE --> I["528 inputs, int8"]
  I --> D0["Dense 528 -> 128, ReLU"] --> D1["Dense 128 -> 64, ReLU"]
  D1 --> BEAT["beat"] & OFF["beat_offset"] & HIT["hit x4"] & MUS["music"]
  BEAT --> S2["stage 2: batch estimate every 0.5 s on 8 s, held grid"]
```"""

MERMAID10 = """```mermaid
flowchart TB
  MIC["mic, I2S 48 kHz int16<br/>512-sample blocks, 93.75/s"] --> FE["frontend.c<br/>12 features"] & MEL["mel flux<br/>16 bands"]
  FE & MEL --> X["28 raw features per block"]
  X --> PA["stage A projection<br/>28 -> 12, float, normalisation folded in"]
  PA --> RA["ring A: 272 blocks x 12"]
  RA --> IA["44 frames: fine 16, mid 16 x 4, coarse 12 x 16<br/>528 inputs, int8"]
  IA --> TA["stage A trunk, int8<br/>Dense 528 -> 128 -> 64, ReLU"]
  TA --> A["A beat, sigmoid<br/>for centre c = newest block - 2"]
  A --> H["history of A beat"]
  H --> FB["fb3: A beat of block j - 3"]
  X --> XB["29 per block: 28 raw + fb3"]
  FB --> XB
  XB --> PB["stage B projection<br/>29 -> 12, float"]
  PB --> RB["ring B: 272 blocks x 12"]
  RB --> IB["528 inputs, int8"]
  IB --> TB["stage B trunk, int8<br/>Dense 528 -> 64 -> 32, ReLU"]
  TB --> BEAT["beat"] & OFF["beat_offset"] & HIT["hit x4<br/>kick / snare / hihat / none"] & MUS["music"]
  BEAT & OFF --> S2["stage 2: batch estimate every 0.5 s on 8 s,<br/>hold the grid until a new fit beats it on the last 4 s"]
  S2 --> G["BPM + beat anchor"]
  MUS --> GATE["gate: median over seconds, 0.7"]
```"""


def page9():
    OUT9.mkdir(parents=True, exist_ok=True)
    pa, pas = load(R / "phaseA_compare.json"), load(R / "phaseA_songeval.json")
    pb, pbs, pbt = load(R / "phaseBC_compare.json"), load(R / "phaseBC_songeval.json"), load(R / "phaseBC_stream.json")
    bn = load(R / "beatnet_eval_v2.json")
    cv2 = load(R / "cv2_songeval.json")
    L = [f"# 9: mel flux and real songs, model d0\n", MERMAID9, ""]

    L.append("## Configuration\n")
    L.append(table(["", ""], [
        ["front end", "`FE_SPEC_VERSION` 2, variant `hicut`, plus `melflux.c` (training: numpy prototype, agrees to 1.2e-5)"],
        ["mel", "1024 Hann ending at the block, rfft power, 40 HTK triangles 150 Hz - 4 kHz merged to 16 bands, log10, positive flux"],
        ["per block", "12 FE + 16 mel flux = 28, projected to 12 (learned, linear, applied per frame)"],
        ["context", "2.87 s, 44 frames x 12 = 528 inputs, ring 272 blocks, lookahead 2 blocks"],
        ["model", "projection 28 -> 12, MLP 528-128-64, heads beat, beat_offset, hit x4, music"],
        ["size", "76,560 MACs per block (336 projection + 76,224 trunk)"],
        ["training pool", "rendered clips, noise, drum loops (file-start grid), melodic loops (phase-free), hand-labelled songs, playlists"],
    ]))

    L.append("\n## Songs\n")
    L.append(table(["", ""], [
        ["source", "songs/likes.csv, 323 tracks, 293 fetched (yt-dlp, duration-matched)"],
        ["labels", "293 by hand in `labeler`: 248 confirmed grids, 38 variable tempo, 7 non-rhythmic"],
        ["label", "constant-tempo grid (bpm, t0) on the file timeline; excluded spans masked"],
        ["pre-annotation", "dataset.songgrid: 12 s locks, grid grown from the longest run, verdicts shown for review"],
        ["playlists", "6 excerpts of 40-90 s per take, hard cut 40% / gap 0.5-2.5 s 20% / crossfade 3-12 s 40%; crossfades masked"],
    ]))

    L.append("\n## Capacity and context, 2 seeds, control n_song (v3 recipe + songs)\n")
    head = ["run", "MACs"] + [s[0] for s in SETS] + ["songs test", "songs val", "lock share, pt"]
    rows = []
    for name, lbl in (("n_song", "control"), ("a1_wide", "wide 224-64"), ("a2_ctx24", "coarse 24 x 16"),
                      ("a3_both", "wide + coarse 24"), ("a4_songs3", "songs rendered 3x")):
        rs = seeds(pa, name)
        rows.append([lbl, f"{rs[0]['macs']:,}"] + set_row(rs) + songs_row(pas, name) + [LOCK.get(name, "-")])
    L.append(table(head, rows))

    L.append("\n## Mel, 2 seeds, rebuilt pool\n")
    rows = []
    for name, lbl in (("b0_ctrl", "control, FE only"), ("b1_flux16", "mel flux 16, no projection"),
                      ("b2_p12", "mel flux 16, projection 12"), ("b3_p16", "mel flux 16, projection 16")):
        rs = seeds(pb, name)
        rows.append([lbl, f"{rs[0]['macs']:,}"] + set_row(rs) + songs_row(pbs, name) + [LOCK.get(name, "-")])
    fx = load(R / "b2fix_compare.json")
    rs = seeds(fx, "b2fix")
    if rs:
        rows.append(["mel flux 16, projection 12, music labels fixed", f"{rs[0]['macs']:,}"] + set_row(rs) + ["-", "-", "-"])
    L.append(table(head, rows))
    L.append("\nLock share for b1 and b2 is against b0; unlabelled songs of 2026-09-28, 189. "
             "b1-b3 (and every mel run before 2026-09-29) trained the music head on labels that were all 0: "
             "`grid5.silent` read loudness from the last column, which mel moved. b2fix is b2 with it fixed.\n")

    L.append("\n## Real songs, 5-fold CV by artist (all 293 labels), d0\n")
    p = cv_pool(cv2, "cv2_d0")
    L.append(table(["songs", "12 s windows", "usable", "tempo exact", "tempo up to octave", "usable per fold"],
                   [[p["songs"], p["windows"], f"{p['usable']*100:.1f}", f"{p['tempo']*100:.1f}",
                     f"{p['octave']*100:.1f}", " / ".join(f"{u*100:.1f}" for u in p["folds"])]]))

    L.append("\n## Against BeatNet (Heydari et al. 2021, model 1), 248 confirmed songs\n")
    L.append(table(["", "BeatNet", "d0"], [
        ["parameters", f"{sum(n for _, n in BEATNET_PARAMS):,}", "76,560"],
        ["MACs per frame", "~405k", "76,560"],
        ["frames per second", "50", "93.75"],
        ["MAC/s", "~20M", "~7.2M"],
        ["weights", "1.6 MB float32", "84 KB int8"],
        ["causal", "online mode only (particle filter)", "yes, 2 blocks lookahead"],
        ["training data", "Ballroom, GTZAN, Beatles, RWC, Rock Corpus", "own corpora + these songs' other folds"],
    ]))
    L.append("")
    rows = []
    for key, lbl in (("ours:auto_cv2_d0", "d0, device audio, held-out fold model"),
                     ("beatnet_offline_clean", "BeatNet offline (DBN), clean audio"),
                     ("beatnet_offline_device", "BeatNet offline (DBN), device audio"),
                     ("beatnet_online_clean", "BeatNet online (PF), clean audio"),
                     ("beatnet_online_device", "BeatNet online (PF), device audio")):
        r = bn.get(key, {})
        rows.append([lbl, mstd([r.get("global_exact")]), mstd([r.get("global_octave")]),
                     mstd([r.get("global_phase_median_ms")], pct=False) + " ms",
                     mstd([r.get("global_phase_within_25ms")]), mstd([r.get("window_usable")]), mstd([r.get("F")])])
    L.append(table(["method", "song grid tempo exact", "up to octave", "phase median", "phase within 25 ms",
                    "12 s windows usable", "F, 70 ms"], rows))
    L.append("\nEvery method through the same stage 2 (tempo.estimate on its events, then one grid fitted "
             "through them). Song grid right (tempo exact, phase within 25 ms at mid-song), 248 songs: "
             "both 73, d0 only 72, BeatNet only 51, neither 52.\n")
    L.append("![beatnet](beatnet.png)\n")
    L.append("![mel](mel.png)\n")
    (OUT9 / "README.md").write_text("\n".join(L))

    fig, ax = new_fig(6.4, 3.0)
    keys = [("ours:auto_cv2_d0", "d0 (device)"), ("beatnet_offline_clean", "BeatNet offline (clean)"),
            ("beatnet_offline_device", "BeatNet offline (device)"), ("beatnet_online_device", "BeatNet online (device)")]
    groups = ["song grid\ntempo exact", "up to octave", "12 s windows\nusable"]
    bar(ax, groups, [(lbl, [bn[k]["global_exact"] * 100, bn[k]["global_octave"] * 100,
                            bn[k]["window_usable"] * 100], COLORS[i]) for i, (k, lbl) in enumerate(keys)],
        100, "% of 248 confirmed songs / windows")
    fig.tight_layout()
    fig.savefig(OUT9 / "beatnet.png")
    plt.close(fig)

    fig, ax = new_fig(6.4, 3.0)
    items = SETS[:6]
    ser = []
    for i, (name, lbl) in enumerate((("b0_ctrl", "FE only"), ("b2_p12", "mel, projection 12"))):
        rs = seeds(pb, name)
        ser.append((lbl, [np.mean([r[t][f] for r in rs]) * 100 for _, t, f in items], COLORS[i]))
    bar(ax, [s[0].replace(" ", "\n", 1) for s in items], ser, 100, "% of test takes, float, 2 seeds")
    fig.tight_layout()
    fig.savefig(OUT9 / "mel.png")
    plt.close(fig)


def page10():
    OUT10.mkdir(parents=True, exist_ok=True)
    pb, pbs, pbt = load(R / "phaseBC_compare.json"), load(R / "phaseBC_songeval.json"), load(R / "phaseBC_stream.json")
    pd, pds, pdt = load(R / "phaseD_compare.json"), load(R / "phaseD_songeval.json"), load(R / "phaseD_stream.json")
    pdo, pdto = load(R / "phaseD_compare_open.json"), load(R / "phaseD_stream_open.json")
    cv, cv2 = load(R / "cv_songeval.json"), load(R / "cv2_songeval.json")
    cvt, cv2t = load(R / "cv_stream.json"), load(R / "cv2_stream.json")
    meta = load(EXPORT / "model_meta.json")
    L = ["# 10: two-stage model g\n", MERMAID10, ""]

    L.append("## Configuration\n")
    sa, sb = meta.get("stage_a", {}), meta.get("stage_b", {})
    L.append(table(["", ""], [
        ["stage A", "d0: projection 28 -> 12, MLP 528-128-64, beat head only in the export"],
        ["stage B", "projection 29 -> 12 (28 features + A beat shifted 3 blocks), MLP 528-64-32, heads beat, beat_offset, hit x4, music"],
        ["feedback", "A beat of block j - 3 as feature 28 of block j: the newest frame is t+2, the newest A output t-1; no added lookahead"],
        ["MACs per block", f"{meta.get('macs_per_block', {}).get('total', 112588):,} (projections 684, A {sa.get('macs_trunk', 75840):,}, B {sb.get('macs_trunk', 36064):,})"],
        ["int8 TFLite", f"A {sa.get('tflite_bytes', 0):,} B, B {sb.get('tflite_bytes', 0):,} B" if sa else "-"],
        ["deployed", "firmware `74cba7b` (2026-09-29)"],
        ["export", f"`{EXPORT}`, A sha256 `{sa.get('sha256', '')[:12]}`, B `{sb.get('sha256', '')[:12]}`" if sa else "-"],
        ["runs", f"`{meta.get('trained_from', {}).get('stage_a_run', 'runs/final_d0_s0')}`, "
                 f"`{meta.get('trained_from', {}).get('stage_b_run', 'runs/final_g_s0')}`, trained on all 293 songs"],
        ["training", "B is trained on A's float activation over every corpus, A frozen; "
                     "final pair on all 293 songs, 7 epochs fixed (the CV median best epoch), no early stopping"],
    ]))

    L.append("\n## Feedback variants, 2 seeds, mel base with playlists (d0 control)\n")
    head = ["run", "added input", "MACs incl. A", "rendered usable", "octave", "melodic grid", "songs test",
            "songs val", "playlists usable", "re-lock median", "old tempo, first 10 s", "never re-locks", "songs usable (stream)"]
    rows = []

    def frow(lbl, add, macs, cmp_, se, st, name, stc="playlists_mel", songs_corpus="songs_mel"):
        rs = seeds(cmp_, name)
        return [lbl, add, macs, mstd([r["clips"]["usable"] for r in rs]), mstd([r["clips"]["octave"] for r in rs]),
                mstd([r["melodic_grid"]["usable"] for r in rs])] + (songs_row(se, name) if se else ["-", "-"]) + \
            stream_row(st, name, stc) + [stream_row(st, name, songs_corpus)[0]]
    rows.append(frow("d0", "-", "76,560", pd, pds, pdt, "d0_ctrl"))
    rows.append(frow("e1, closed loop", "own beat, shift 3", "76,572", pd, pds, pdt, "e1_fb"))
    rows.append(frow("e2, closed loop", "own beat (trained on e1's), shift 3", "76,572", pd, pds, pdt, "e2_fb"))
    rows.append(frow("e1 fed d0 (two-stage)", "d0 beat, shift 3", "153,132", pdo, None, pdto, "e1_fb"))
    rows.append(frow("f1", "d0 stream-tracker state", "153,168", pd, pds, pdt, "f1_st"))
    L.append(table(head, rows))
    L.append("\nPhase C (FE-only base, cascade from n_song): c1 with A's unshifted beat had 2 blocks of added "
             "lookahead; its gain is not comparable. Every run on this page except the final pair trained its "
             "music head on labels that were all 0 (see results/9); they compare with each other like for like. "
             "The final pair (`runs/final3_*`) has it fixed.\n")

    L.append("\n## 5-fold CV by artist, balanced by genre, one model per fold\n")
    rows = []
    for tag, se, st, corpus, labels in (("cv", cv, cvt, "{c}_f{k}_mel", "172"), ("cv2", cv2, cv2t, "{c}_v2f{k}_mel", "293")):
        for cfg, lbl, macs in (("d0", "d0", "76,560"), ("g", "g, stage B 64-32", "112,588"), ("h", "h, stage B 128-64", "153,132")):
            if f"runs/{tag}_{cfg}_f0_s0" not in se:
                continue
            p = cv_pool(se, f"{tag}_{cfg}")
            pl = cv_stream(st, f"{tag}_{cfg}", tag, corpus.replace("{c}", "playlists"))
            sg = cv_stream(st, f"{tag}_{cfg}", tag, corpus.replace("{c}", "songs"))
            rows.append([labels, lbl, macs, p["songs"], p["windows"], f"{p['usable']*100:.1f}",
                         " / ".join(f"{u*100:.1f}" for u in p["folds"]), f"{p['tempo']*100:.1f}",
                         f"{p['octave']*100:.1f}", f"{sg.get('usable', 0)*100:.1f}", f"{pl.get('usable', 0)*100:.1f}",
                         f"{pl.get('relock_median_s', 0):.1f} s", f"{pl.get('never', 0)*100:.1f}"])
    L.append(table(["labels", "model", "MACs", "songs", "windows", "usable", "usable per fold", "tempo exact",
                    "octave", "songs usable (stream)", "playlists usable", "re-lock median", "never re-locks"], rows))
    L.append("\n![folds](folds.png)\n")

    L.append("\n## Standard test sets, mean of the 5 fold models\n")
    rows = []
    for tag in ("cv", "cv2"):
        by = cv_sets(R / f"{tag}_compare.json")
        for cfg in ("d0", "g", "h"):
            key = f"{tag}_{cfg}"
            if key in by:
                rows.append([tag, cfg] + set_row(by[key]))
    L.append(table(["CV", "model"] + [s[0] for s in SETS], rows))

    v3c = load(R / "v3_compare_mel.json").get("runs/m_cand_s0")
    v3s = [load(R / f"v3_songeval_f{k}.json").get("runs/m_cand_s0", {}).get("test") for k in range(5)]
    v3t = load(R / "v3_stream.json")
    if v3c and all(v3s) and meta:
        w = np.array([r["windows"] for r in v3s], float)
        v3u = float(np.sum([r["usable"] for r in v3s] * w) / w.sum())
        v3p = [v3t[f"runs/m_cand_s0:playlists_v2f{k}_mel:test:hold8"] for k in range(5)]
        g2 = cv_pool(cv2, "cv2_g")
        g2p = cv_stream(cv2t, "cv2_g", "cv2", "playlists_v2f{k}_mel")
        mt = meta["metrics"]
        L.append("\n## Against v3 (music v3, `runs/m_cand_s0`, deployed before this model)\n")
        L.append(table(["", "v3", "two-stage"], [
            ["real songs, 12 s windows usable (293 songs; v3 never trained on songs, two-stage cross-validated)",
             f"{v3u*100:.1f}", f"{g2['usable']*100:.1f}"],
            ["playlists usable, hold 8 s", f"{np.mean([r['usable'] for r in v3p])*100:.1f}", f"{g2p['usable']*100:.1f}"],
            ["re-lock after a track change, median", f"{np.mean([r['relock_median_s'] for r in v3p]):.1f} s",
             f"{g2p['relock_median_s']:.1f} s"],
            ["never re-locks", f"{np.mean([r['never'] for r in v3p])*100:.1f}", f"{g2p['never']*100:.1f}"],
            ["rendered clips usable (v3 float, two-stage final int8)", f"{v3c['clips']['usable']*100:.1f}",
             f"{mt['clips']['int8']['usable']*100:.1f}"],
            ["drum loops tempo", f"{v3c['loops']['tempo']*100:.1f}", f"{mt['loops']['int8']['tempo']*100:.1f}"],
            ["drum loops grid usable", f"{v3c['loops_grid']['usable']*100:.1f}", f"{mt['loops_grid']['int8']['usable']*100:.1f}"],
            ["melodic loops tempo", f"{v3c['melodic']['tempo']*100:.1f}", f"{mt['melodic']['int8']['tempo']*100:.1f}"],
            ["melodic loops grid usable", f"{v3c['melodic_grid']['usable']*100:.1f}", f"{mt['melodic_grid']['int8']['usable']*100:.1f}"],
            ["MACs per block", "76,224", f"{meta['macs_per_block']['total']:,}"],
        ]))

    gate = load(EXPORT / "gate.json")
    if gate:
        v3g = {"real loops": 84.9, "melodic": 18.7, "rendered": 79.0, "non-music": 88.3, "silence": 100.0}
        L.append("\n## Music gate, int8, val, per-take median of stage B music\n")
        rows = []
        for gname in ("real loops", "melodic", "rendered", "songs", "non-music", "silence"):
            r = gate[gname]
            kind = "rejected" if gname in ("non-music", "silence") else "kept"
            rows.append([gname, kind, r["takes"], f"{v3g[gname]:.1f}" if gname in v3g else "-",
                         f"{r['0.7']*100:.1f}", f"{r['0.8']*100:.1f}"])
        L.append(table(["group", "", "takes", "v3 at 0.7", "two-stage at 0.7", "two-stage at 0.8"], rows))

    if meta:
        L.append("\n## Export, float against int8, test splits, 461-block window\n")
        rows = []
        for name, r in meta["metrics"].items():
            key = "usable" if "usable" in r["float32"] else "tempo"
            rows.append([name, key, f"{r['float32'][key]*100:.1f}", f"{r['int8'][key]*100:.1f}",
                         f"{r['correlation']:.4f}"])
        L.append(table(["test set", "metric", "float", "int8", "beat correlation"], rows))
    L.append("\n![trace](trace.png)\n")
    (OUT10 / "README.md").write_text("\n".join(L))

    fig, ax = new_fig(6.4, 3.0)
    groups = [f"{t} fold {k}" for t in ("cv", "cv2") for k in range(5)]
    ser = []
    for i, cfg in enumerate(("d0", "g")):
        v = []
        for tag, se in (("cv", cv), ("cv2", cv2)):
            v += [se[f"runs/{tag}_{cfg}_f{k}_s0"]["test"]["usable"] * 100 for k in range(5)]
        ser.append((cfg, v, COLORS[i]))
    bar(ax, [g.replace(" fold ", "\n") for g in groups], ser, 100, "% of held-out 12 s windows usable")
    fig.tight_layout()
    fig.savefig(OUT10 / "folds.png")
    plt.close(fig)

    g = load(EXPORT / "golden" / "golden.json")
    if g:
        lab = load(Path("songs/labels") / f"{g['song']}.json")
        n = g["blocks"]
        t = np.arange(n) * 512 / 48000
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.6), dpi=160, sharex=True)
        fig.patch.set_facecolor("#ffffff")
        T = 60.0 / lab["bpm"]
        beats = lab["t0_s"] + np.arange(np.ceil((g["start_s"] - lab["t0_s"]) / T),
                                        np.ceil((g["start_s"] + g["seconds"] - lab["t0_s"]) / T)) * T - g["start_s"]
        for ax, key, lbl, col in ((axes[0], "a_beat_int8_dequant", "stage A beat, int8", COLORS[0]),
                                  (axes[1], "b_beat", "stage B beat, int8", COLORS[1])):
            ax.plot(t, g[key], color=col, linewidth=0.8, label=lbl)
            for b in beats:
                ax.axvline(b, color=INK2, linewidth=0.5, linestyle=(0, (3, 3)))
            ax.set_ylim(0, 1.05)
            style(ax, ylabel="activation")
            ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper right")
        axes[1].set_xlabel("s from boot (golden excerpt); dashed: hand-labelled beats", fontsize=7, color=INK2)
        fig.tight_layout()
        fig.savefig(OUT10 / "trace.png")
        plt.close(fig)


def main():
    page9()
    page10()
    print("written", OUT9, OUT10)


if __name__ == "__main__":
    main()
