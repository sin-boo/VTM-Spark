"""Dev stand-in for /api when the full desk backend is not on :8765."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
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


def _packs() -> list[Path]:
    if not CHARS.is_dir():
        return []
    found: list[Path] = []
    for item in CHARS.iterdir():
        if item.is_file() and item.suffix.lower() == ".vtm":
            found.append(item)
        elif item.is_dir():
            found.extend(p for p in item.iterdir() if p.suffix.lower() == ".vtm")
    return sorted((p for p in found if p.stem not in _hidden), key=lambda p: p.name.lower())


def _manifest(path: Path) -> dict:
    try:
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("manifest.json").decode("utf-8"))
    except Exception:
        return {}


def _preview_bytes(path: Path) -> bytes | None:
    still = CHARS / path.stem / "preview.png"
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
        ident = path.stem
        cards.append(
            {
                "id": ident,
                "name": str(raw.get("name") or ident),
                "path": f"characters/{path.name}",
                "preview_url": f"/api/characters/{ident}/preview",
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
    raw = _manifest(cur) if cur else {}
    ident = cur.stem if cur else ""
    name = str(raw.get("name") or ident)
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
        p = self.path.split("?")[0]
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
        p = self.path.split("?")[0]
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
