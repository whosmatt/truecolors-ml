"""Summarise a manifest: what the corpus actually contains after filtering.

    python -m dataset.stats data/manifest.jsonl
"""

import argparse
import collections
import json
from pathlib import Path


def load(path: Path):
    with path.open() as fh:
        header = json.loads(fh.readline())
        return header, [json.loads(line) for line in fh]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path, nargs="?", default=Path("data/manifest.jsonl"))
    a = ap.parse_args()
    header, rows = load(a.manifest)

    print(f"built {header['built']}  Live index {header['index']['live_version']}  {len(rows)} rows\n")

    print(f"{'class':8} {'kind':10} {'total':>7} {'kept':>7} {'pack':>6} {'auto':>6} {'both':>6}")
    by = collections.Counter((r["class"], r["kind"]) for r in rows)
    for (cls, kind), n in sorted(by.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        sel = [r for r in rows if r["class"] == cls and r["kind"] == kind]
        kept = sum(not r["discard"] for r in sel)
        src = collections.Counter(r["label_source"] for r in sel)
        print(f"{str(cls):8} {kind:10} {n:7} {kept:7} {src['pack']:6} {src['auto']:6} {src['both']:6}")

    unreadable = [r for r in rows if not r.get("readable", True)]
    if unreadable:
        import os
        ext = collections.Counter(os.path.splitext(r["name"])[1].lower() for r in unreadable)
        print(f"\nunreadable {len(unreadable)} ({len(unreadable)/len(rows):.1%}) by extension: {dict(ext.most_common(5))}")
        lost_pack = sum(r["label_source"] in ("pack", "both") for r in unreadable)
        pack_total = sum(r["label_source"] in ("pack", "both") for r in rows)
        print(f"  of which pack-tagged: {lost_pack} of {pack_total} pack-tagged rows ({lost_pack/max(pack_total,1):.0%})")

    print("\ndiscards by reason:")
    for reason, n in collections.Counter(
        r["discard_reason"] for r in rows if r["discard"]
    ).most_common():
        print(f"  {str(reason):16} {n}")

    loops = [r for r in rows if r["kind"] == "loop"]
    with_bpm = [r for r in loops if r.get("bpm")]
    print(f"\nloops {len(loops)}, tempo recovered {len(with_bpm)} ({len(with_bpm)/max(len(loops),1):.1%})")
    print("  by source:", dict(collections.Counter(r["bpm_source"] for r in with_bpm)))
    print("  by bars:  ", dict(sorted(collections.Counter(r["bars"] for r in with_bpm).items())))

    one = [r for r in rows if r["kind"] == "one_shot" and not r["discard"]]
    if one:
        off = sorted(r["attack_offset_ms"] for r in one if r["attack_offset_ms"] is not None)
        lead = sorted(r["lead_silence_ms"] for r in one if r["lead_silence_ms"] is not None)
        pct = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]
        print(f"\nkept one-shots {len(one)}")
        print(f"  attack offset ms  p50 {pct(off,.5):.2f}  p90 {pct(off,.9):.2f}  max {off[-1]:.2f}")
        print(f"  lead silence ms   p50 {pct(lead,.5):.2f}  p90 {pct(lead,.9):.2f}  >1ms {sum(x>1 for x in lead)/len(lead):.1%}")

    print("\npack-vs-auto label agreement (files carrying both kinds of tag):")
    from .ableton import CLASS_TAGS, KEY_CKEY, KEY_KEYW, Library

    lib = Library()
    for cls in CLASS_TAGS:
        pack, auto = lib.class_members(cls)
        # Live only auto-classifies files without pack metadata, so agreement is
        # only measurable on the overlap.
        ids = ",".join(str(i) for i in pack) or "0"
        judged = {
            r[0]
            for r in lib._q(
                f"select distinct file_id from metadata where key = {KEY_CKEY} and file_id in ({ids})"
            )
        }
        agree = len(pack & auto)
        print(f"  {cls:7} {agree:5}/{len(judged):5} = {agree/max(len(judged),1):5.1%}"
              f"   (pack-tagged {len(pack)}, auto-classified {len(auto)})")

    sr = collections.Counter(r["samplerate"] for r in rows if r.get("samplerate"))
    print("\nsample rates:", dict(sr.most_common(6)))
    dup = collections.Counter(r["head_sha256"] for r in rows if r.get("head_sha256"))
    ndup = sum(n - 1 for n in dup.values() if n > 1)
    print(f"duplicate content (head hash): {ndup} rows share a hash with an earlier one")


if __name__ == "__main__":
    main()
