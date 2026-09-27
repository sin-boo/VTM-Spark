"""Dev stand-in for /api when the full desk backend is not on :8765."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from email.message import Message
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import quote, unquote
import zipfile

ROOT = Path(__file__).resolve().parent.parent
CHARS = ROOT / "characters"
DIT = ROOT / "models" / "dit"
_MIN_CKPT = 1_000_000
_CKPT_SUFFIX = {".pt", ".pth", ".ckpt"}


def _dit_files() -> list[Path]:
    if not DIT.is_dir():
        return []
    found: list[Path] = []
    try:
        entries = list(DIT.iterdir())
    except OSError:
        return []
    nested: list[Path] = []
    for path in entries:
        if path.is_dir() and not path.name.startswith("."):
            nested.append(path)
            continue
        if _is_dit(path):
            found.append(path)
    for child in nested:
        try:
            kids = list(child.iterdir())
        except OSError:
            continue
        for path in kids:
            if _is_dit(path):
                found.append(path)
    return sorted(found, key=lambda p: p.name.lower())


def _is_dit(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    if path.suffix.lower() not in _CKPT_SUFFIX:
        return False
    try:
        return path.stat().st_size >= _MIN_CKPT
    except OSError:
        return False


def _checkpoints() -> list[dict]:
    cards = []
    for path in _dit_files():
        rel = path.relative_to(ROOT).as_posix()
        cards.append({"label": path.stem, "path": rel, "source": "local"})
    return cards


def _session() -> dict:
    path = ROOT / "models" / "session.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


# Imported packs live in a temp folder so the mock never writes to characters/.
IMPORTS = Path(tempfile.gettempdir()) / "vtm_mock_imports"
# MOCK_EXPORT_SAVE=1 pretends the native Save dialog exists; =cancel simulates Cancel.
# Unset, /export-save 404s and the UI falls back to the GET download.
_EXPORT_SAVE = os.environ.get("MOCK_EXPORT_SAVE", "").strip().lower()
# Set to a checkpoint file name to make packs made with another model show model_match=false.
_MOCK_CHECKPOINT = os.environ.get("MOCK_CHECKPOINT", "").strip()


def _packs() -> list[Path]:
    found: list[Path] = []
    for root in (CHARS, IMPORTS):
        if not root.is_dir():
            continue
        for item in root.iterdir():
            if item.is_file() and item.suffix.lower() == ".vtm":
                found.append(item)
            elif item.is_dir():
                found.extend(p for p in item.iterdir() if p.suffix.lower() == ".vtm")
    return sorted((p for p in found if p.stem not in _hidden), key=lambda p: p.name.lower())


def _pack(ident: str) -> Path | None:
    for cand in _packs():
        if cand.stem == ident:
            return cand
    return None


def _manifest(path: Path) -> dict:
    try:
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("manifest.json").decode("utf-8"))
    except Exception:
        return {}


def _thumb_bytes(path: Path) -> bytes | None:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.read("thumb.png")
    except Exception:
        return _preview_bytes(path)


# id -> {name, author, license, description} edits made through /meta.
_meta: dict[str, dict] = {}


def _pack_meta(path: Path) -> dict:
    raw = _manifest(path)
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    out = {
        "name": str(raw.get("name") or path.stem),
        "author": str(meta.get("author") or ""),
        "license": str(meta.get("license") or ""),
        "description": str(meta.get("description") or ""),
    }
    out.update(_meta.get(path.stem, {}))
    return out


def _info(path: Path) -> dict:
    raw = _manifest(path)
    meta = _pack_meta(path)
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    size = int(raw.get("image_size") or model.get("image_size") or 512)
    checkpoint = str(model.get("checkpoint") or "")
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except Exception:
        names = set()
    stamp = str(raw.get("created_at") or "2026-09-20T22:23:47Z")
    shape = model.get("latent_shape") or [4, size // 8, size // 8]
    return {
        "id": path.stem,
        "name": meta["name"],
        "version": int(raw.get("version") or 1),
        "created_at": stamp,
        "updated_at": str(raw.get("updated_at") or stamp),
        "author": meta["author"],
        "license": meta["license"],
        "description": meta["description"],
        "model": {"checkpoint": checkpoint, "image_size": size, "latent_shape": list(shape)},
        "model_match": not _MOCK_CHECKPOINT or checkpoint == _MOCK_CHECKPOINT,
        "includes": {
            "pose_keys": sum(1 for n in names if n.startswith("poses/")) or (6 if names else 0),
            "blendshapes": any("blend" in n or "shapes" in n for n in names),
            "hair": any("hair" in n for n in names),
            "skeleton": any("skeleton" in n or n == "fit.json" for n in names),
            "travel_box": "fit.json" in names,
            "source_image": any(n.startswith("source") for n in names),
        },
        "size_bytes": path.stat().st_size,
    }


def _multipart_file(content_type: str, body: bytes) -> tuple[str, bytes] | None:
    """Return (filename, bytes) of the multipart field named ``file``."""
    head = Message()
    head["Content-Type"] = content_type
    boundary = head.get_param("boundary")
    if not isinstance(boundary, str) or not boundary:
        return None
    for chunk in body.split(b"--" + boundary.encode()):
        headers, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        part = Message()
        part["Content-Disposition"] = next(
            (
                line.split(b":", 1)[1].strip().decode("utf-8", "replace")
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-disposition:")
            ),
            "",
        )
        if part.get_param("name", header="content-disposition") != "file":
            continue
        name = part.get_param("filename", header="content-disposition")
        return (str(name) if name else "import.vtm"), data.removesuffix(b"\r\n")
    return None


def _preview_bytes(path: Path) -> bytes | None:
    # Cache next to where the pack lives so imports never touch characters/.
    base = CHARS if path.is_relative_to(CHARS) else IMPORTS
    still = base / path.stem / "preview.png"
    if still.is_file():
        return still.read_bytes()
    try:
        with zipfile.ZipFile(path) as zf:
            data = zf.read("preview.png")
        still.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(data)
        return data
    except Exception:
        return None


def _cards() -> list[dict]:
    cards = []
    for path in _packs():
        raw = _manifest(path)
        meta = _pack_meta(path)
        ident = path.stem
        stamp = int(path.stat().st_mtime_ns)
        cards.append(
            {
                "id": ident,
                "name": meta["name"],
                "path": f"characters/{path.name}",
                "preview_url": f"/api/characters/{ident}/preview?v={stamp}",
                "thumb_url": f"/api/characters/{ident}/thumb?v={stamp}",
                "author": meta["author"],
                "version": int(raw.get("version") or 1),
            }
        )
    return cards


_hidden: set[str] = set()
_current_id: str | None = None


def _current() -> Path | None:
    packs = _packs()
    wanted = _current_id
    if wanted is None:
        raw = str(_session().get("character_path") or "").strip()
        wanted = Path(raw).stem if raw else ""
    if not wanted:
        return None
    for path in packs:
        if path.stem == wanted:
            return path
    return None


ready = {
    "ready": True,
    "running": False,
    "error": "",
    "awaiting": "",
    "suggested": "",
    "progress": 1,
    "progress_label": "Ready",
    "stages": {
        "model": {"state": "done", "progress": 1, "label": "Ready"},
        "character": {"state": "done", "progress": 1, "label": "Ready"},
        "lab": {"state": "done", "progress": 1, "label": "Ready"},
    },
}


_settings: dict = {
    "interpolate": True,
    "hold_last": True,
    "compile_model": False,
    "show_mesh": False,
    "show_hair": False,
    "show_outline": False,
    "show_brows": False,
    "show_eyes": False,
    "show_nose": False,
    "show_mouth": False,
    "show_iris_overlay": False,
    "show_skeleton": False,
    "mirror": True,
    "travel_box": {},
}


def status() -> dict:
    cur = _current()
    ident = cur.stem if cur else ""
    name = _pack_meta(cur)["name"] if cur else ""
    return {
        "state": "idle",
        "message": "Idle",
        "checkpoint": next(
            (c["label"] for c in _checkpoints() if c["path"] == str(_session().get("checkpoint") or "").replace("\\", "/")),
            (_checkpoints()[0]["label"] if _checkpoints() else ""),
        ),
        "device": "cuda",
        "error": "",
        "model_ready": True,
        "ref_ready": bool(cur),
        "streaming": False,
        "paused": False,
        "tracking": False,
        "busy": False,
        "fast_warming": False,
        "steps": 1,
        "pose_cfg": 1,
        "id_cfg": 1,
        "frame_blend": 0.58,
        "inbetweens": 1,
        "max_fps": 0,
        "interpolate": True,
        "hold_last": True,
        "track_fps": 2,
        "drive_pose": True,
        "show_mesh": False,
        "show_hair": False,
        "show_outline": False,
        "show_brows": False,
        "show_eyes": False,
        "show_nose": False,
        "show_mouth": False,
        "show_iris_overlay": False,
        "show_skeleton": False,
        "mirror": True,
        "use_iris": True,
        "use_body": True,
        "fast_mode": True,
        "compile_model": False,
        "batch2": False,
        "auto_sync_track": True,
        "gen_fps": 0,
        "show_fps": 0,
        "timing": "",
        "reference_name": name,
        "reference_path": f"characters/{cur.name}" if cur else "",
        "character_id": ident,
        "character_name": name,
        "camera_index": 0,
        "track_message": "",
        "body_label": "",
        "progress": 0,
        "progress_label": "",
        "progress_kind": "",
        "compile_on": False,
        "compile_status": "",
        "compile_detail": "",
        "virtual_cam": False,
        "virtual_cam_device": "VTM Noble Cam",
        "virtual_cam_error": "",
        "virtual_cam_width": 0,
        "virtual_cam_height": 0,
        "pose_frozen": False,
        **_settings,
    }


lab = {
    "online": False,
    "live": False,
    "error": "Track Lab is not running",
    "source": "ifm",
    "camera_index": 0,
    "cameras": [{"index": 0, "name": "Integrated Camera"}],
    "feel": {},
    "weights": {},
    "calib": {},
    "ifm": {"port": 49983, "primary": "192.168.0.2"},
}


class H(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        return

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = unquote(self.path.split("?")[0])
        if p in ("/api/boot", "/api/status"):
            self._send(200, ready if "boot" in p else status())
        elif p == "/api/checkpoints":
            self._send(200, _checkpoints())
        elif p == "/api/models/catalog":
            offers = []
            try:
                import sys

                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from backend.model_download import hub_catalog_offers

                offers = hub_catalog_offers()
            except Exception:
                offers = []
            self._send(200, {"offers": offers})
        elif p == "/api/reference/default":
            self._send(200, {"exists": "true", "path": "models/refs/gigi.png"})
        elif p == "/api/lab/status":
            self._send(200, lab)
        elif p == "/api/cameras":
            self._send(200, {"cameras": lab["cameras"], "preferred": 0})
        elif p == "/api/characters":
            self._send(200, {"characters": _cards()})
        elif p.startswith("/api/characters/") and p.endswith("/thumb"):
            path = _pack(p[len("/api/characters/") : -len("/thumb")])
            data = _thumb_bytes(path) if path else None
            if data:
                self._send(200, data, "image/png")
            else:
                self._send(404, {"detail": "Character not found"})
        elif p.startswith("/api/characters/") and p.endswith("/info"):
            path = _pack(p[len("/api/characters/") : -len("/info")])
            if path is None:
                self._send(404, {"detail": "Character not found"})
            else:
                self._send(200, _info(path))
        elif p.startswith("/api/characters/") and p.endswith("/export"):
            path = _pack(p[len("/api/characters/") : -len("/export")])
            if path is None:
                self._send(404, {"detail": "Character not found"})
                return
            data = path.read_bytes()
            fname = f"{_pack_meta(path)['name']}.vtm"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Disposition",
                f"attachment; filename=\"{path.name}\"; filename*=UTF-8''{quote(fname)}",
            )
            self.end_headers()
            self.wfile.write(data)
        elif p.startswith("/api/characters/") and p.endswith("/preview"):
            ident = p[len("/api/characters/") : -len("/preview")]
            packs = _packs()
            path = None
            if ident == "current":
                path = _current()
            else:
                for cand in packs:
                    if cand.stem == ident:
                        path = cand
                        break
            data = _preview_bytes(path) if path else None
            if data:
                self._send(200, data, "image/png")
            elif ident == "current":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)
        elif p == "/api/ws":
            self.send_error(400)
        elif not p.startswith("/api/"):
            page = (
                "<!doctype html><title>Mock API</title>"
                "<body style='margin:0;background:#2b2b2b;color:#cfcfcf;"
                "font:14px Segoe UI,sans-serif;padding:24px'>"
                "This is the Vite mock API on port 8765, not the operator desk. "
                "Open <a href='http://127.0.0.1:5173'>http://127.0.0.1:5173</a>, "
                "or close this process and launch VTM Noble."
            )
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(200, {})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode() or "{}")
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def do_POST(self):
        p = unquote(self.path.split("?")[0])
        if p == "/api/settings":
            patch = self._read_json()
            for key, value in patch.items():
                _settings[key] = value
            self._send(200, status())
            return
        if p == "/api/lab/command":
            body = self._read_json()
            op = str(body.get("op") or "")
            payload = body.get("body") if isinstance(body.get("body"), dict) else body
            if op == "set_feel" and isinstance(payload, dict):
                feel = lab.get("feel") if isinstance(lab.get("feel"), dict) else {}
                feel.update(payload)
                lab["feel"] = feel
            self._send(200, {"ok": True, "status": lab, "online": lab.get("online")})
            return
        if p == "/api/characters/meta":
            body = self._read_json()
            path = _pack(str(body.get("id") or "").strip())
            if path is None:
                self._send(400, {"detail": "Character not found"})
                return
            if body.get("name") is not None and not str(body["name"]).strip():
                self._send(400, {"detail": "Name cannot be empty"})
                return
            edits = _meta.setdefault(path.stem, {})
            for key in ("name", "author", "license", "description"):
                if body.get(key) is not None:
                    edits[key] = str(body[key]).strip()
            card = next(c for c in _cards() if c["id"] == path.stem)
            self._send(200, {"ok": True, "character": card, "characters": _cards()})
            return
        if p == "/api/characters/reveal":
            body = self._read_json()
            path = _pack(str(body.get("id") or "").strip())
            if path is None:
                self._send(400, {"detail": "Character not found"})
                return
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{path.resolve()}"')
            self._send(200, {"ok": True, "path": str(path)})
            return
        if p == "/api/characters/export-save":
            body = self._read_json()
            path = _pack(str(body.get("id") or "").strip())
            if not _EXPORT_SAVE:
                self._send(404, {"detail": "Not Found"})
            elif path is None:
                self._send(400, {"detail": "Character not found"})
            elif _EXPORT_SAVE == "cancel":
                self._send(200, {"ok": False, "cancelled": True})
            else:
                dest = Path.home() / "Downloads" / f"{_pack_meta(path)['name']}.vtm"
                self._send(200, {"ok": True, "path": str(dest)})
            return
        if p == "/api/characters/add":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            got = _multipart_file(self.headers.get("Content-Type") or "", raw)
            if got is None:
                self._send(400, {"detail": "No file uploaded"})
                return
            fname, data = got
            if not fname.lower().endswith(".vtm"):
                self._send(400, {"detail": "Not a .vtm character file"})
                return
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    json.loads(zf.read("manifest.json").decode("utf-8"))
            except Exception:
                self._send(400, {"detail": "This file is not a valid VTM character pack"})
                return
            IMPORTS.mkdir(parents=True, exist_ok=True)
            stem = Path(fname).stem or "import"
            dest = IMPORTS / f"{stem}.vtm"
            n = 2
            while _pack(dest.stem) is not None or dest.exists():
                dest = IMPORTS / f"{stem}-{n}.vtm"
                n += 1
            dest.write_bytes(data)
            _hidden.discard(dest.stem)
            card = next(c for c in _cards() if c["id"] == dest.stem)
            self._send(200, {"ok": True, "character": card, "status": status()})
            return
        if p == "/api/characters/remove":
            global _current_id
            body = self._read_json()
            ident = str(body.get("id") or "").strip()
            cur = _current()
            if ident:
                _hidden.add(ident)
            if cur is not None and cur.stem == ident:
                _current_id = ""
            self._send(200, {"ok": True, "status": status(), "characters": _cards()})
            return
        if p == "/api/models/download":
            self._send(200, {"status": "idle", "message": "Mock — no download", "progress": 0})
            return
        if p == "/api/reload":
            self._send(200, {"ok": True, "reloading": False})
            return
        self.do_GET()


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8765), H).serve_forever()
