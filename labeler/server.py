"""Standard-library HTTP backend for the beat grid labeler.

Routes:
    GET  /                      the UI (static/)
    GET  /api/songs             index rows with status ok, plus auto verdict and label status
    GET  /api/song/<id>         {row, auto|null, label|null}
    GET  /api/auto/<id>         the auto analysis (404 when not analysed)
    GET  /api/label/<id>        the human label (404 when absent)
    POST /api/label             {id, status, bpm, t0_s, sections, note} -> saved label
    POST /api/eval              {id, bpm, t0_s} -> evaluate() of that grid
    GET  /audio/<id>            the song's audio, with Range support
"""

import argparse
import json
import math
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

STATIC = Path(__file__).parent / "static"
REPO = Path(__file__).resolve().parent.parent
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STATUSES = {"confirmed", "variable_tempo", "non_rhythmic", "skipped"}
AUDIO_TYPES = {".opus": "audio/ogg", ".ogg": "audio/ogg", ".mp3": "audio/mpeg",
               ".m4a": "audio/mp4", ".wav": "audio/wav", ".flac": "audio/flac",
               ".webm": "audio/webm"}
CHUNK = 256 * 1024


class Store:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._index: list[dict] = []
        self._index_key = None
        self._auto_cache: dict[str, tuple[float, dict]] = {}

    def index(self) -> list[dict]:
        # The downloader rewrites index.jsonl whole; a read mid-rewrite may be short
        # or missing, so the last good parse is kept when nothing parses.
        p = self.root / "index.jsonl"
        try:
            st = p.stat()
            key = (st.st_mtime_ns, st.st_size)
            with self._lock:
                if key == self._index_key:
                    return self._index
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return self._index
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and isinstance(r.get("id"), str) and ID_RE.match(r["id"]):
                rows.append(r)
        if rows or not self._index:
            with self._lock:
                self._index, self._index_key = rows, key
        return self._index

    def row(self, sid: str) -> dict | None:
        return next((r for r in self.index() if r["id"] == sid), None)

    def _read_json(self, p: Path) -> dict | None:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def auto(self, sid: str) -> dict | None:
        p = self.root / "auto" / f"{sid}.json"
        try:
            mt = p.stat().st_mtime_ns
        except OSError:
            return None
        with self._lock:
            hit = self._auto_cache.get(sid)
        if hit and hit[0] == mt:
            return hit[1]
        d = self._read_json(p)
        if d is not None:
            with self._lock:
                self._auto_cache[sid] = (mt, d)
        return d

    def label(self, sid: str) -> dict | None:
        return self._read_json(self.root / "labels" / f"{sid}.json")

    def write_label(self, sid: str, data: dict):
        d = self.root / "labels"
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{sid}.", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(finite(data), f, indent=1, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o644)
            os.replace(tmp, d / f"{sid}.json")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def audio_path(self, sid: str) -> Path | None:
        r = self.row(sid)
        cands = []
        if r and isinstance(r.get("audio"), str):
            cands.append(self.root / r["audio"])
        cands.append(self.root / "audio" / f"{sid}.opus")
        root = self.root.resolve()
        for c in cands:
            try:
                c = c.resolve()
            except OSError:
                continue
            if c.is_file() and root in c.parents:
                return c
        return None


def finite(o):
    """NaN/inf -> None: songgrid writes them via json.dumps, browsers reject them."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def evaluate_grid(auto: dict, bpm: float, t0: float) -> dict:
    """-> evaluate() output, or {"error": ...} when the import or the call fails."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from dataset.songgrid import evaluate
    except Exception as e:  # numpy/frontend missing or broken: report, don't crash
        return {"error": f"import dataset.songgrid failed: {type(e).__name__}: {e}"}
    try:
        return evaluate(auto, float(bpm), float(t0))
    except Exception as e:
        traceback.print_exc()
        return {"error": f"evaluate failed: {type(e).__name__}: {e}"}


def song_summary(store: Store, r: dict) -> dict:
    a = store.auto(r["id"])
    lab = store.label(r["id"])
    v = (a or {}).get("verdict") or {}
    fit = (a or {}).get("fit") or {}
    return {
        "id": r["id"], "artist": r.get("artist"), "title": r.get("title"),
        "duration_s": r.get("duration_s"), "spotify_tempo": r.get("spotify_tempo"),
        "analysed": a is not None, "auto_verdict": v.get("auto"),
        "needs_review": v.get("needs_review"), "auto_bpm": fit.get("bpm"),
        "label_status": (lab or {}).get("status"),
        "has_audio": store.audio_path(r["id"]) is not None,
    }


class Handler(BaseHTTPRequestHandler):
    store: Store
    server_version = "labeler/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if os.environ.get("LABELER_QUIET"):
            return
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(finite(obj), allow_nan=False, default=str).encode(),
                   "application/json; charset=utf-8")

    def _err(self, code: int, msg: str):
        self._json({"error": msg}, code)

    def _body(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n <= 0 or n > 8 * 1024 * 1024:
            return None
        try:
            d = json.loads(self.rfile.read(n))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return d if isinstance(d, dict) else None

    def _id(self, s: str) -> str | None:
        s = unquote(s)
        return s if ID_RE.match(s) else None

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._err(500, f"{type(e).__name__}: {e}")
            except Exception:
                pass

    def do_POST(self):
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._err(500, f"{type(e).__name__}: {e}")
            except Exception:
                pass

    def _get(self):
        path = urlparse(self.path).path
        st = self.store
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/api/songs":
            rows = [r for r in st.index() if r.get("status") == "ok"]
            return self._json({"songs": [song_summary(st, r) for r in rows]})
        m = re.match(r"^/api/(song|auto|label)/([^/]+)$", path)
        if m:
            kind, sid = m.group(1), self._id(m.group(2))
            if sid is None:
                return self._err(400, "bad id")
            if kind == "auto":
                a = st.auto(sid)
                return self._json(a) if a is not None else self._err(404, "not analysed")
            if kind == "label":
                lab = st.label(sid)
                return self._json(lab) if lab is not None else self._err(404, "no label")
            r = st.row(sid)
            if r is None:
                return self._err(404, "unknown id")
            return self._json({"row": r, "auto": st.auto(sid), "label": st.label(sid)})
        m = re.match(r"^/audio/([^/]+?)(?:\.[a-z0-9]+)?$", path)
        if m:
            sid = self._id(m.group(1))
            p = st.audio_path(sid) if sid else None
            if p is None:
                return self._err(404, "no audio")
            return self._file(p, AUDIO_TYPES.get(p.suffix.lower(), "application/octet-stream"))
        self._err(404, "not found")

    def _static(self, rel: str):
        p = (STATIC / unquote(rel)).resolve()
        if STATIC.resolve() not in p.parents or not p.is_file():
            return self._err(404, "not found")
        ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}.get(
            p.suffix, mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self._send(200, p.read_bytes(), ctype + "; charset=utf-8")

    def _file(self, p: Path, ctype: str):
        size = p.stat().st_size
        start, end = 0, size - 1
        code = 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                return self._range_fail(size)
            if m.group(1) == "":
                n = int(m.group(2))
                if n == 0:
                    return self._range_fail(size)
                start = max(size - n, 0)
            else:
                start = int(m.group(1))
                if m.group(2) != "":
                    end = min(int(m.group(2)), size - 1)
            if start >= size or start > end:
                return self._range_fail(size)
            code = 206
        n = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(n))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            while n > 0:
                b = f.read(min(CHUNK, n))
                if not b:
                    break
                self.wfile.write(b)
                n -= len(b)

    def _range_fail(self, size: int):
        self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}",
                                            "Accept-Ranges": "bytes"})

    def _post(self):
        path = urlparse(self.path).path
        body = self._body()
        if body is None:
            return self._err(400, "expected a JSON object body")
        sid = self._id(str(body.get("id", "")))
        if sid is None:
            return self._err(400, "bad id")
        if path == "/api/eval":
            try:
                bpm, t0 = float(body["bpm"]), float(body["t0_s"])
            except (KeyError, TypeError, ValueError):
                return self._err(400, "bpm and t0_s must be numbers")
            if not (1.0 <= bpm <= 1000.0):
                return self._err(400, "bpm out of range")
            a = self.store.auto(sid)
            if a is None:
                return self._err(404, "not analysed")
            res = evaluate_grid(a, bpm, t0)
            return self._json(res, 500 if "error" in res else 200)
        if path == "/api/label":
            return self._save_label(sid, body)
        self._err(404, "not found")

    def _save_label(self, sid: str, body: dict):
        st = self.store
        if st.row(sid) is None:
            return self._err(404, "unknown id")
        status = body.get("status")
        if status not in STATUSES:
            return self._err(400, f"status must be one of {sorted(STATUSES)}")

        def num(k):
            v = body.get(k)
            try:
                return None if v is None else float(v)
            except (TypeError, ValueError):
                return None

        bpm, t0 = num("bpm"), num("t0_s")
        if status == "confirmed" and (bpm is None or t0 is None or not 1.0 <= bpm <= 1000.0):
            return self._err(400, "confirmed needs numeric bpm and t0_s")
        secs = []
        for s in body.get("sections") or []:
            try:
                secs.append({"start_s": float(s["start_s"]), "end_s": float(s["end_s"]),
                             "include": bool(s.get("include", True))})
            except (KeyError, TypeError, ValueError):
                return self._err(400, "sections need start_s, end_s, include")
        a = st.auto(sid)
        ev = None
        if a is not None and bpm is not None and t0 is not None and 1.0 <= bpm <= 1000.0:
            ev = evaluate_grid(a, bpm, t0)
        label = {
            "id": sid, "status": status, "bpm": bpm, "t0_s": t0, "sections": secs,
            "auto_verdict": (a or {}).get("verdict"), "auto_fit": (a or {}).get("fit"),
            "auto_analysed": (a or {}).get("analysed"), "auto_model": (a or {}).get("model"),
            "fe_spec_version": (a or {}).get("fe_spec_version"),
            "eval": ev, "note": str(body.get("note") or ""),
            "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        st.write_label(sid, label)
        self._json(label)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m labeler", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", type=Path, default=Path("songs"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if not (a.songs / "index.jsonl").exists():
        print(f"warning: {a.songs / 'index.jsonl'} not found (yet)", file=sys.stderr)
    Handler.store = Store(a.songs)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"labeler on http://{a.host}:{srv.server_address[1]}/  songs={a.songs.resolve()}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
