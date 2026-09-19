"""Read-only access to Live 12's sample index.

DB access READ ONLY
"""

import base64
import os
import re
import sqlite3
import struct
import urllib.parse
from pathlib import Path

DB_PATH = Path(
    "/mnt/c/Users/matt/AppData/Local/Ableton/Live Database/Live-files-12300.db"
)

# fourcc ints, as stored in files.file_type
FT_WAV, FT_AIFF, FT_MP3 = 2002875949, 1634297446, 1836069677
AUDIO_TYPES = (FT_WAV, FT_AIFF)  # mp3 excluded: encoder delay shifts onsets

KEY_KEYW = 1264941431  # pack / user keywords
KEY_CKEY = 1129014649  # Live's auto-classifier

# Union of both keys per class. Snare and clap are one class; they overlap
# acoustically and the detector does not need to separate them.
CLASS_TAGS = {
    "kick": ("Drums|Kick",),
    "snare": (
        "Drums|Snare|Snare Hit",
        "Drums|Snare|Rim",
        "Drums|Snare|Snare Articulation",
        "Drums|Snare",
        "Drums|Clap",
        "Drums|Percussion|Snap",
    ),
    "hihat": (
        "Drums|Hihat|Closed Hihat",
        "Drums|Hihat|Open Hihat",
        "Drums|Hihat|Pedal Hihat",
        "Drums|Hihat",
    ),
    # Not a detection target. Carried so rendered grooves can voice their
    # percussion parts, which are 59% of the notes in the factory drum clips and
    # are exactly the distractors a kick/snare detector must not fire on.
    "perc": (
        "Drums|Tom|Low Tom", "Drums|Tom|Mid Tom", "Drums|Tom|High Tom",
        "Drums|Cymbal|Ride", "Drums|Cymbal|Crash", "Drums|Cymbal|Splash",
        "Drums|Cymbal|Misc Cymbal", "Drums|Cymbal",
        "Drums|Percussion|Electronic", "Drums|Percussion|Shaker",
        "Drums|Percussion|Conga", "Drums|Percussion|Wood",
        "Drums|Percussion|Bell", "Drums|Percussion|Misc Percussion",
        "Drums|Percussion|Cowbell", "Drums|Percussion|Tambourine",
        "Drums|Percussion|Bongo", "Drums|Percussion|Chime",
        "Drums|Percussion|Triangle", "Drums|Percussion|Timbale",
        "Drums|Percussion|Djembe", "Drums|Percussion|Tabla",
        "Drums|Percussion|Gong", "Drums|Percussion|Timpani",
    ),
}

DETECTION_CLASSES = ("kick", "snare", "hihat")
KIND_TAGS = {"one_shot": ("Type|One Shot",), "loop": ("Type|Loop", "Drums|Drum Loop")}


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    uri = "file:" + urllib.parse.quote(str(path)) + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


class Library:
    """File tree, tags and embeddings from one Live index."""

    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.con = connect(self.path)
        self.version = self.con.execute("select version from version").fetchone()[0]
        self._tree = {
            fid: (pid, name)
            for fid, pid, name in self.con.execute(
                "select file_id, parent_id, name from files"
            )
        }

    def _q(self, sql, args=()):
        return self.con.execute(sql, args).fetchall()

    def parts(self, file_id: int) -> list[str]:
        out, fid = [], file_id
        while fid in self._tree:
            pid, name = self._tree[fid]
            if name:
                out.append(name)
            fid = pid
            if not fid:
                break
        return list(reversed(out))

    def path_of(self, file_id: int) -> str:
        """Windows path from the folder chain, translated to its WSL mount."""
        p = self.parts(file_id)
        drive = p[0].rstrip("\\").rstrip(":").lower()
        return "/mnt/" + drive + "/" + "/".join(p[1:])

    def name_of(self, file_id: int) -> str:
        return self._tree[file_id][1]

    def folders_of(self, file_id: int, depth: int = 3) -> str:
        return " / ".join(reversed(self.parts(file_id)[:-1][-depth:]))

    def tagged(self, value: str, key: int | None = None) -> set[int]:
        keys = (key,) if key else (KEY_KEYW, KEY_CKEY)
        ph = ",".join("?" * len(keys))
        rows = self._q(
            f"""select f.file_id from files f
                join metadata m on m.file_id = f.file_id
                join metadata_values v on v.id = m.value_id
                where m.key in ({ph}) and v.value = ?
                  and f.file_type in {AUDIO_TYPES}""",
            (*keys, value),
        )
        return {r[0] for r in rows}

    def class_members(self, cls: str) -> tuple[set[int], set[int]]:
        """(pack-tagged, auto-tagged) file ids for one class."""
        pack, auto = set(), set()
        for tag in CLASS_TAGS[cls]:
            pack |= self.tagged(tag, KEY_KEYW)
            auto |= self.tagged(tag, KEY_CKEY)
        return pack, auto

    def kind_members(self, kind: str) -> set[int]:
        out = set()
        for tag in KIND_TAGS[kind]:
            out |= self.tagged(tag)
        return out

    def embeddings(self) -> dict[int, bytes]:
        """file_id -> 64 float32, Live's sound-similarity vector.

        Blob layout: uint32 version, uint32 dim, uint32 reserved, then dim floats.
        """
        out = {}
        for fid, data in self._q("select file_id, data from fe_values"):
            if not data or len(data) < 12:
                continue
            _, dim, _ = struct.unpack("<3I", data[:12])
            want = 12 + dim * 4
            if dim != 64 or len(data) < want:
                continue
            out[fid] = data[12:want]
        return out

    def fingerprint(self) -> dict:
        """Identifies the index a manifest was built from."""
        st = self.path.stat()
        return {
            "path": str(self.path),
            "live_version": self.version,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        }


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
