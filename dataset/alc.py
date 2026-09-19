"""Parse Ableton .alc drum clips into note events with exact times.

Factory .alc clips are gzipped XML: a Drum Rack plus the MIDI that plays it.
They are the only source here with true onset ground truth, and the timing is
humanised rather than quantised (kick/snare deviate from the 16th grid by a
median of 1.0 ms but p90 of 23 ms), which is the error scale the detector has to
resolve.

Two things are not obvious from the XML:

- `<KeyTrack Id="N">` carries an attribute, so a literal `<KeyTrack>` match
  finds nothing.
- A pad's MIDI note is `128 - BranchInfo/ReceivingNote`, and the pad's *name*
  is what identifies the instrument. General MIDI pitch is not reliable: in
  "Backbeat Funky Swing" note 38 is a rim and 41 is a tom.
"""

import argparse
import gzip
import re
from dataclasses import dataclass

# Checked against the pad names across all 985 factory drum clips. Order matters:
# "Hihat" must beat "Hi Tom", and the snare group absorbs claps, rims and snaps.
_PAD_RULES = (
    ("hihat", ("hihat", "hi-hat", "hi hat", "openhat", "closedhat", "hats", "hat ")),
    ("kick", ("kick", "bassdrum", "bass drum", "808 bd", " bd ")),
    ("snare", ("snare", "clap", "rimshot", "rim ", "sidestick", "side stick", "snap")),
)

_KEYTRACK = re.compile(rb"<KeyTrack\b(.*?)</KeyTrack>", re.S)
_MIDIKEY = re.compile(rb'<MidiKey Value="(-?\d+)"')
_NOTE = re.compile(
    rb'<MidiNoteEvent Time="(-?[\d.]+)"[^>]*?Velocity="([\d.]+)"[^>]*?/>'
)
_TEMPO = re.compile(rb'<Tempo>\s*<LomId Value="0" />\s*<Manual Value="([\d.]+)"')
_END = re.compile(rb'<CurrentEnd Value="([\d.]+)"')
_BRANCH = re.compile(
    rb"<DrumBranch\b.*?<EffectiveName Value=\"([^\"]*)\".*?<ReceivingNote Value=\"(\d+)\"",
    re.S,
)


def classify_pad(name: str) -> str:
    low = " " + name.lower() + " "
    for cls, keys in _PAD_RULES:
        if any(k in low for k in keys):
            return cls
    return "other"


@dataclass
class Note:
    beat: float
    midi: int
    velocity: float
    pad: str
    cls: str


@dataclass
class Clip:
    name: str
    tempo: float
    beats: float
    notes: list[Note]

    @property
    def duration_s(self) -> float:
        return self.beats * 60.0 / self.tempo

    def time_of(self, note: Note) -> float:
        return note.beat * 60.0 / self.tempo


def parse(path: str, name: str = "") -> Clip | None:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    t, e = _TEMPO.search(raw), _END.search(raw)
    if not t or not e:
        return None
    tempo, beats = float(t.group(1)), float(e.group(1))
    if tempo <= 0 or beats <= 0:
        return None

    pads = {
        128 - int(note): nm.decode("latin1", "replace")
        for nm, note in _BRANCH.findall(raw)
    }
    notes = []
    for block in _KEYTRACK.findall(raw):
        k = _MIDIKEY.search(block)
        if not k:
            continue
        midi = int(k.group(1))
        pad = pads.get(midi, "")
        cls = classify_pad(pad) if pad else "other"
        for beat, vel in _NOTE.findall(block):
            b = float(beat)
            if 0.0 <= b < beats:  # notes can sit outside the loop bounds
                notes.append(Note(b, midi, float(vel), pad, cls))
    notes.sort(key=lambda n: n.beat)
    return Clip(name or path.rsplit("/", 1)[-1], tempo, beats, notes)


def drum_clips(lib) -> list[int]:
    return [
        r[0]
        for r in lib._q(
            """select distinct f.file_id from files f
               join metadata m on m.file_id = f.file_id
               join metadata_values v on v.id = m.value_id
               where m.key in (1264941431, 1129014649)
                 and v.value like 'Clips|Drum Clip%' and f.file_type = 1634493229"""
        )
    ]


def main():
    import collections

    from .ableton import Library

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    lib = Library()
    ids = drum_clips(lib)[: a.limit or None]
    cls_n, pad_unmapped, tempos = collections.Counter(), collections.Counter(), []
    beats = notes = 0
    bad = 0
    for fid in ids:
        c = parse(lib.path_of(fid), lib.name_of(fid))
        if c is None:
            bad += 1
            continue
        tempos.append(c.tempo)
        beats += c.beats
        notes += len(c.notes)
        for n in c.notes:
            cls_n[n.cls] += 1
            if n.cls == "other":
                pad_unmapped[n.pad or f"<no pad {n.midi}>"] += 1
    print(f"clips {len(ids)}, unparsed {bad}, notes {notes}, beats {beats:.0f}")
    print(f"tempo range {min(tempos):.0f}-{max(tempos):.0f}, median {sorted(tempos)[len(tempos)//2]:.0f}")
    print("\nnotes by class:")
    for c, n in cls_n.most_common():
        print(f"  {c:7} {n:7} ({n/notes:5.1%})")
    print("\ntop unmapped pads:")
    for p, n in pad_unmapped.most_common(12):
        print(f"  {n:6}  {p}")


if __name__ == "__main__":
    main()
