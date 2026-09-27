"""How well the automatic song filters and fits agree with the hand labels.

    python -m dataset.songcheck

Verdict confusion (auto rows, human columns), then for songs confirmed by hand
with an auto fit: tempo agreement (exact, octave) and the phase difference of
the two grids at the song's middle, in ms. Without labels, the auto fits are
checked against Spotify's tempo instead.
"""

import json
from collections import Counter

import numpy as np

from .songgrid import SONGS


def phase_diff_ms(bpm_a, t0_a, bpm_b, t0_b, at_s):
    """Distance from grid a's beat nearest `at_s` to grid b's nearest beat."""
    Ta, Tb = 60.0 / bpm_a, 60.0 / bpm_b
    ga = t0_a + round((at_s - t0_a) / Ta) * Ta
    gb = t0_b + round((ga - t0_b) / Tb) * Tb
    return (ga - gb) * 1000.0


def main():
    index = {r["id"]: r for r in map(json.loads, (SONGS / "index.jsonl").read_text().splitlines())}
    auto = {p.stem: json.loads(p.read_text()) for p in (SONGS / "auto").glob("*.json")}
    labels = {p.stem: json.loads(p.read_text()) for p in (SONGS / "labels").glob("*.json")}
    print(f"{len(index)} songs, {sum(r['status'] == 'ok' for r in index.values())} downloaded, "
          f"{len(auto)} analysed, {len(labels)} labelled")
    print("auto verdicts:", dict(Counter(a["verdict"]["auto"] for a in auto.values())))

    ex = oc = n = 0
    for k, a in auto.items():
        sp = index[k].get("spotify_tempo") or 0
        if a["fit"] and sp:
            r = a["fit"]["bpm"] / sp
            n += 1
            ex += abs(r - 1) < 0.02
            oc += any(abs(r / h - 1) < 0.02 for h in (1, 2, 0.5, 1.5, 2 / 3))
    if n:
        print(f"auto fit vs spotify tempo ({n} fits): exact {ex/n:.1%}, up to octave {oc/n:.1%}")

    if not labels:
        return
    conf = Counter((auto[k]["verdict"]["auto"] if k in auto else "none", l["status"])
                   for k, l in labels.items())
    rows = sorted({r for r, _ in conf})
    cols = sorted({c for _, c in conf})
    print("\n" + "auto \\ human".ljust(18) + "".join(c[:14].rjust(15) for c in cols))
    for r in rows:
        print(r.ljust(18) + "".join(str(conf[(r, c)]).rjust(15) for c in cols))

    d = []
    for k, l in labels.items():
        f = (auto.get(k) or {}).get("fit")
        if l["status"] != "confirmed" or not f:
            continue
        ratio = f["bpm"] / l["bpm"]
        mid = auto[k]["duration_s"] / 2
        d.append((abs(ratio - 1) < 0.001, abs(ratio - 1) < 0.001 or abs(ratio - 2) < 0.002
                  or abs(ratio - 0.5) < 0.001,
                  phase_diff_ms(f["bpm"], f["t0_s"], l["bpm"], l["t0_s"], mid),
                  (f["bpm"] - l["bpm"]) / l["bpm"] * 1e6))
    if d:
        e = np.array(d, dtype=float)
        same = e[e[:, 0] > 0]
        print(f"\nconfirmed with an auto fit: {len(e)}; tempo within 0.1%: {e[:, 0].mean():.1%}, "
              f"up to octave: {e[:, 1].mean():.1%}")
        if len(same):
            print(f"  same tempo: |phase diff| median {np.median(np.abs(same[:, 2])):.1f} ms, "
                  f"p90 {np.percentile(np.abs(same[:, 2]), 90):.1f} ms; "
                  f"|tempo diff| median {np.median(np.abs(same[:, 3])):.0f} ppm")


if __name__ == "__main__":
    main()
