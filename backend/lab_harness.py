"""HTTP client from VTM Noble to the Track Lab harness."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_BASE = os.environ.get("VTM_LAB_HARNESS", "http://127.0.0.1:8780/harness")
LAB_PROTOCOL_PREFIX = "track_lab.harness"
NUM_KEYPOINTS = 37
# Camera open / HRNet fit can take well over the 0.6s status poll timeout.
COMMAND_TIMEOUT = 8.0
SLOW_OPS: dict[str, float] = {
    "start": 180.0,
    "stop": 30.0,
    "track": 180.0,
    "set_source": 60.0,
    "reset": 30.0,
}
_FRAME_INTO_STATUS = (
    "weights",
    "points",
    "skeleton",
    "hair",
    "keypoints",
    "calib",
    "image_wh",
    "head",
    "blink",
    "iris",
    "iris_method",
    "iris_cam",
    "look",
    "hair_method",
    "point_offsets",
)


def looks_like_lab(payload: object) -> bool:
    """True when a JSON body is a Track Lab harness packet, not some other HTTP app."""
    if not isinstance(payload, dict):
        return False
    protocol = str(payload.get("protocol") or "")
    if protocol.startswith(LAB_PROTOCOL_PREFIX):
        return True
    kind = str(payload.get("type") or "")
    if kind in {"status", "frame", "ack"}:
        return True
    return str(payload.get("service") or "") == "track_lab"


def occupied_error(base: str = DEFAULT_BASE) -> str:
    host = base.replace("http://", "").replace("https://", "").split("/", 1)[0]
    port = host.rsplit(":", 1)[-1] if ":" in host else "8780"
    return (
        f"Port {port} is already in use by another app. "
        "Close that app, then start Track Lab."
    )


DEFAULT_FEEL: dict[str, float] = {
    "response": 0.65,
    "smoothing": 0.48,
    "mouth": 0.50,
    "use_visemes": 1.0,
    "show_face": 1.0,
    "show_skeleton": 1.0,
    "show_hair": 1.0,
    "show_ids": 0.0,
    "hair_pin": 0.7,
    "max_yaw": 1.0,
    "max_roll": 1.0,
    "max_pitch_up": 1.0,
    "max_pitch_down": 1.0,
    "max_look_x": 1.0,
    "max_look_y": 1.0,
    "gaze_gain": 1.0,
    "gaze_smooth": 0.28,
}


def offline_status(error: str = "Track Lab is not running") -> dict[str, Any]:
    return {
        "type": "status",
        "online": False,
        "ok": False,
        "live": False,
        "loaded": False,
        "error": str(error or "Track Lab is not running"),
        "feel": dict(DEFAULT_FEEL),
        "commands": [],
        "cameras": [],
        "camera_index": 0,
        "source": "camera",
        "weights": {},
        "points": [],
        "skeleton": [],
        "hair": [],
        "hair_method": "none",
        "keypoints": [],
        "calib": {},
        "ifm": {},
        "point_offsets": [],
        "shapes": {},
        "presets": [],
        "active": "",
        "iris": [],
        "iris_method": "none",
        "iris_cam": [],
        "look": None,
        "blink": {"l": 0.0, "r": 0.0},
        "head": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    }


def _log(msg: str) -> None:
    print(f"[lab-harness] {msg}", flush=True)


def _origin(base: str) -> str:
    text = str(base).rstrip("/")
    if text.endswith("/harness"):
        return text[: -len("/harness")]
    return text.rsplit("/", 1)[0] if "/" in text else text


def _snippet(raw: bytes | str | None, limit: int = 180) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _http_body(exc: urllib.error.HTTPError) -> str:
    try:
        return _snippet(exc.read())
    except Exception:
        return ""


class LabHarness:
    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 0.6) -> None:
        self.base = str(base).rstrip("/")
        self.timeout = float(timeout)
        self._last_note = ""

    def _note(self, msg: str) -> None:
        if msg == self._last_note:
            return
        self._last_note = msg
        _log(msg)

    def status(self, *, merge_frame: bool = True) -> dict[str, Any]:
        try:
            payload, frame = self._status_and_frame(merge_frame=merge_frame)
        except json.JSONDecodeError as exc:
            error = occupied_error(self.base)
            self._note(f"status {self.base}/status not JSON: {exc}")
            return offline_status(error)
        except Exception as exc:
            error = _short_error(exc)
            self._note(f"status {self.base}/status failed: {error}")
            return offline_status(error)
        if payload is None:
            self._note(f"status {self.base}/status empty body")
            return offline_status("Track Lab returned an empty status")
        if not looks_like_lab(payload):
            keys = ",".join(sorted(str(k) for k in payload.keys())[:8])
            self._note(
                f"status {self.base}/status not a harness packet "
                f"(type={payload.get('type')!r} protocol={payload.get('protocol')!r} keys={keys})"
            )
            return offline_status(occupied_error(self.base))
        payload["online"] = True
        payload.setdefault("ok", True)
        payload.setdefault("feel", dict(DEFAULT_FEEL))
        if "loaded" not in payload:
            payload["loaded"] = True
        feel = dict(DEFAULT_FEEL)
        raw = payload.get("feel")
        if isinstance(raw, dict):
            for key, value in DEFAULT_FEEL.items():
                try:
                    feel[key] = float(raw.get(key, value))
                except (TypeError, ValueError):
                    feel[key] = value
        payload["feel"] = feel
        if merge_frame and isinstance(frame, dict):
            merge_frame_into_status(payload, frame)
        self._note(
            f"online {self.base} protocol={payload.get('protocol') or '—'} "
            f"source={payload.get('source') or '—'} live={bool(payload.get('live'))} "
            f"loaded={bool(payload.get('loaded'))}"
        )
        return payload

    def _status_and_frame(
        self, *, merge_frame: bool
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Pull /status and /frame together so live meters do not wait twice."""
        if not merge_frame:
            return self._get("/status"), None
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            status_job = pool.submit(self._get, "/status")
            frame_job = pool.submit(self.frame)
            return status_job.result(), frame_job.result()

    def frame(self) -> dict[str, Any] | None:
        try:
            return self._get("/frame")
        except Exception as exc:
            self._note(f"frame {self.base}/frame failed: {_short_error(exc)}")
            return None

    def handshake(self, *, attempts: int = 1, delay: float = 0.0) -> dict[str, Any]:
        """GET status + ping. Does not start tracking. Does not wait for torch.

        A live /status is enough to hook in. Ping can time out while the CUDA
        worker is still loading; that must not look like 'Track Lab did not start'.
        """
        last = offline_status("Track Lab is not running")
        tries = max(1, int(attempts))
        wait = max(0.0, float(delay))
        for _ in range(tries):
            last = self.status(merge_frame=False)
            if last.get("online"):
                ping = self.command("ping")
                ping_ok = bool(ping.get("ok")) and bool(ping.get("online", True))
                err = str(ping.get("error") or "")
                last["handshake"] = True
                last["online"] = True
                last["ok"] = True
                last["error"] = ""
                if ping_ok and not err:
                    self._note(f"handshake ok {self.base}")
                else:
                    self._note(f"handshake host-only {self.base} ping={err or '—'}")
                return last
            last["handshake"] = False
            if wait:
                time.sleep(wait)
        last["handshake"] = False
        self._note(f"handshake failed {self.base}: {last.get('error') or 'offline'}")
        return last

    def command(self, op: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        timeout = 2.0 if op == "ping" else float(SLOW_OPS.get(op, COMMAND_TIMEOUT))
        _log(f"command {self.base}/command op={op} body={body or {}} timeout={timeout}")
        try:
            payload = self._post("/command", {"op": op, "body": body or {}}, timeout=timeout)
        except Exception as exc:
            error = _short_error(exc, command=True)
            _log(f"command op={op} failed: {error}")
            recovered = self._recover_command(op, error)
            if recovered is not None:
                return recovered
            self._note(f"offline after command {op}: {error}")
            out = offline_status(error)
            out["type"] = "ack"
            out["ok"] = False
            return out
        if not isinstance(payload, dict):
            _log(f"command op={op} empty reply")
            return {"type": "ack", "ok": False, "error": "empty reply", "online": True}
        payload["online"] = True
        nested = payload.get("status") if isinstance(payload.get("status"), dict) else {}
        _log(
            f"command op={op} ok={payload.get('ok')} error={payload.get('error') or '—'} "
            f"source={nested.get('source') or payload.get('source') or '—'}"
        )
        return payload

    def _recover_command(self, op: str, error: str) -> dict[str, Any] | None:
        """If a slow op timed out after the lab actually finished, don't lie 'offline'."""
        if op not in SLOW_OPS:
            return None
        probe = self.status()
        if not probe.get("online"):
            return None
        live = bool(probe.get("live"))
        ready = bool(probe.get("ready"))
        if op == "start":
            ok = live
        elif op == "stop":
            ok = not live
        elif op == "track":
            ok = ready
        else:
            ok = True
        _log(f"command op={op} recovered online live={live} ok={ok}")
        return {
            "type": "ack",
            "ok": ok,
            "online": True,
            "error": "" if ok else error,
            "status": probe,
        }

    def put_source(self, path: str) -> dict[str, Any]:
        """Load a still into Track Lab (harness op, or /api/source on an older lab)."""
        ack = self.command("set_source", {"path": str(path)})
        err = str(ack.get("error") or "").lower()
        if ack.get("ok") is False and "unknown" in err:
            _log("set_source missing — posting /api/source")
            return self._post_source_file(path)
        return ack

    def _post_source_file(self, path: str) -> dict[str, Any]:
        src = Path(path)
        url = _origin(self.base).rstrip("/") + "/api/source"
        req = urllib.request.Request(
            url,
            data=src.read_bytes(),
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "x-filename": src.name,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=SLOW_OPS["set_source"]) as res:
                raw = res.read()
        except Exception as exc:
            error = _short_error(exc, command=True)
            out = offline_status(error)
            out["type"] = "ack"
            out["ok"] = False
            return out
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
        err = str(parsed.get("error") or "") if isinstance(parsed, dict) else "empty reply"
        return {
            "type": "ack",
            "ok": not err,
            "online": True,
            "error": err,
            "status": parsed if isinstance(parsed, dict) else {},
        }

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any] | None:
        url = self.base + path
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else timeout) as res:
                raw = res.read()
        except urllib.error.HTTPError as exc:
            self._on_http_error("GET", url, exc, noisy=False)
            raise
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        url = self.base + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        wait = self.timeout if timeout is None else timeout
        try:
            with urllib.request.urlopen(req, timeout=wait) as res:
                raw = res.read()
        except urllib.error.HTTPError as exc:
            self._on_http_error("POST", url, exc, noisy=True)
            raise
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}

    def _on_http_error(self, method: str, url: str, exc: urllib.error.HTTPError, *, noisy: bool) -> None:
        body = _http_body(exc)
        line = f"{method} {url} -> HTTP {exc.code}"
        if body:
            line += f" body={body!r}"
        if exc.code == 404:
            line += " | " + _health_probe(url)
        if noisy:
            _log(line)
        else:
            self._note(line)


def _health_probe(url: str) -> str:
    root = url.split("/harness", 1)[0] if "/harness" in url else _origin(url)
    health = root.rstrip("/") + "/api/health"
    try:
        with urllib.request.urlopen(health, timeout=0.4) as res:
            raw = res.read(256)
            return f"GET {health} -> HTTP {res.status} body={_snippet(raw)!r}"
    except urllib.error.HTTPError as health_exc:
        return f"GET {health} -> HTTP {health_exc.code} (no /api/health — old Track Lab process?)"
    except Exception as health_exc:
        return f"GET {health} failed: {health_exc}"


def merge_frame_into_status(payload: dict[str, Any], frame: dict[str, Any]) -> None:
    """Live overlay / meters live on /frame, not the last published /status."""
    for key in _FRAME_INTO_STATUS:
        if key in frame and frame[key] is not None:
            payload[key] = frame[key]
    if frame.get("live") is not None:
        payload["live"] = bool(frame.get("live"))
    if "loaded" in frame:
        payload["loaded"] = bool(frame.get("loaded"))
    err = frame.get("error")
    if err and (bool(frame.get("live")) or bool(payload.get("live"))):
        payload["error"] = str(err)


def frame_image_wh(frame: dict[str, Any] | None) -> tuple[int, int] | None:
    """Lab character_px canvas from a harness frame, or None if missing."""
    if not isinstance(frame, dict):
        return None
    wh = frame.get("image_wh") if isinstance(frame.get("image_wh"), (list, tuple)) else ()
    w = h = 0
    if len(wh) >= 2:
        try:
            w, h = int(wh[0] or 0), int(wh[1] or 0)
        except (TypeError, ValueError):
            w = h = 0
    if w <= 0 or h <= 0:
        try:
            w = int(frame.get("width") or 0)
            h = int(frame.get("height") or 0)
        except (TypeError, ValueError):
            w = h = 0
    if w <= 0 or h <= 0:
        return None
    return w, h


def _frame_wh(
    frame: dict[str, Any],
    *,
    width: int = 0,
    height: int = 0,
) -> tuple[int, int]:
    """Convert character_px with the lab canvas, not the desk jpeg size.

    Desk ``_last_image`` is only a fallback when the harness omitted ``image_wh``.
    """
    lab = frame_image_wh(frame)
    if lab is not None:
        return lab
    return max(int(width) or 0, 1), max(int(height) or 0, 1)


def overlay_from_frame(
    frame: dict[str, Any] | None,
    *,
    width: int = 0,
    height: int = 0,
) -> np.ndarray | None:
    """Harness character_px keypoints → (37, 4) norm_crop for the cel overlay.

    ``image_wh`` on the frame is the source canvas. Desk jpeg size is fallback only.
    """
    if not isinstance(frame, dict):
        return None
    rows = frame.get("keypoints")
    if not isinstance(rows, list) or not rows:
        return None
    from .pose_controller import slot_of

    k = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
    visible = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = slot_of(row.get("i"))
        if idx < 0:
            idx = slot_of(row.get("ref") or row.get("name") or row.get("legacy"))
        if idx < 0 or idx >= NUM_KEYPOINTS:
            continue
        try:
            k[idx, 0] = float(row.get("x", 0.0))
            k[idx, 1] = float(row.get("y", 0.0))
            k[idx, 2] = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        on = bool(row.get("visible")) or float(k[idx, 2]) >= 0.5
        k[idx, 3] = 1.0 if on else 0.0
        if on:
            visible += 1
    if visible < 1:
        return None
    w, h = _frame_wh(frame, width=width, height=height)
    out = k.copy()
    out[:, 0] = k[:, 0] / float(w) * 2.0 - 1.0
    out[:, 1] = k[:, 1] / float(h) * 2.0 - 1.0
    return out


def hair_from_frame(
    frame: dict[str, Any] | None,
    *,
    width: int = 0,
    height: int = 0,
) -> list[dict[str, Any]] | None:
    """Harness character_px hair polygons → norm_crop for the cel overlay.

    Track Lab already follows / yaws the three hair parts. The desk must copy
    those polygons, not rebuild them with a different follow.
    Missing ``hair`` returns None (keep the last good mesh). An empty list
    means the lab sent no parts this frame.
    """
    if not isinstance(frame, dict) or "hair" not in frame:
        return None
    raw = frame.get("hair")
    if not isinstance(raw, list):
        return None
    w, h = _frame_wh(frame, width=width, height=height)
    sx = 2.0 / float(w)
    sy = 2.0 / float(h)
    out: list[dict[str, Any]] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        cls = str(seg.get("class") or "")
        poly = seg.get("polygon") or []
        if not cls or not isinstance(poly, list) or len(poly) < 3:
            continue
        pts: list[list[float]] = []
        for vertex in poly:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            try:
                pts.append([float(vertex[0]) * sx - 1.0, float(vertex[1]) * sy - 1.0])
            except (TypeError, ValueError):
                continue
        if len(pts) < 3:
            continue
        out.append({"class": cls, "polygon": pts})
    return out


def _short_error(exc: BaseException, *, command: bool = False) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        url = str(getattr(exc, "url", "") or "")
        if exc.code == 404:
            where = f" ({exc.code} GET {url})" if url else ""
            return "Track Lab is on an old build. Restart it to load the harness." + where
        return f"Track Lab HTTP {exc.code}" + (f" {url}" if url else "")
    text = str(exc).lower()
    timed = isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        reason_text = str(reason or exc).lower()
        if isinstance(reason, ConnectionRefusedError) or "refused" in reason_text:
            return "Track Lab is not running"
        timed = timed or "timed out" in reason_text or "timeout" in reason_text
        if not timed:
            return str(reason or exc)
    if timed:
        return "Track Lab timed out" if command else "Track Lab is not running"
    return str(exc)


def tracker_loaded(packet: dict[str, Any] | None) -> bool:
    """True when the torch worker is attached. Missing ``loaded`` means an old lab."""
    if not isinstance(packet, dict) or not packet.get("online"):
        return False
    if "loaded" in packet:
        return bool(packet.get("loaded"))
    return True


def wait_loaded(
    *,
    timeout: float = 90.0,
    on_wait: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Poll status until the worker is attached. Host handshake is already done."""
    end = time.monotonic() + max(0.0, float(timeout))
    start = time.monotonic()
    span = max(0.001, float(timeout))
    last = lab.status()
    if tracker_loaded(last):
        if on_wait is not None:
            on_wait(1.0)
        return last
    while time.monotonic() < end:
        last = lab.status()
        if tracker_loaded(last):
            if on_wait is not None:
                on_wait(1.0)
            return last
        if on_wait is not None:
            on_wait(min(1.0, (time.monotonic() - start) / span))
        time.sleep(0.25)
    if on_wait is not None:
        on_wait(min(1.0, (time.monotonic() - start) / span))
    return last


lab = LabHarness()
