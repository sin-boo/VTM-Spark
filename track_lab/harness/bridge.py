"""Localhost JSON-line IPC between the torch-free host and the tracker worker.

The host binds 8780 and speaks harness HTTP. The worker loads torch, then
connects here and pushes packed frames. Commands are forwarded over the same
socket. This is not a public API.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from .hub import hub
from .pack import warming_frame, warming_status
from .protocol import ack

ENV_PORT = "TRACK_LAB_IPC_PORT"
NO_WORKER = "TRACK_LAB_NO_WORKER"
HOST = "127.0.0.1"
CMD_WAIT = 180.0
GEN_WAIT = 300.0
JPEG_WAIT = 4.0
LIVE_OPS = frozenset({"start", "stop", "calibrate", "track"})


def _log(msg: str) -> None:
    print(f"[harness-ipc] {msg}", flush=True)


def encode(msg: dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":"), default=str).encode("utf-8") + b"\n"


def send(sock: socket.socket, msg: dict[str, Any], lock: threading.Lock | None = None) -> None:
    payload = encode(msg)
    if lock is None:
        sock.sendall(payload)
        return
    with lock:
        sock.sendall(payload)


def iter_messages(sock: socket.socket):
    buf = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(msg, dict):
                yield msg


def connect(port: int, timeout: float = 8.0) -> socket.socket:
    sock = socket.create_connection((HOST, int(port)), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    return sock


def publish_warming(*, clients: int | None = None) -> None:
    n = hub.clients if clients is None else int(clients)
    hub.publish(warming_status(clients=n))
    hub.publish(warming_frame(clients=n))


def _publish_ack_status(msg: dict[str, Any]) -> None:
    """Copy a command reply onto the host status slot.

    The worker sends status and then a frame. The frame used to replace the
    unsent status on the IPC mailbox, so GET /status kept saying camera after
    set_input. The ack still carries the real snapshot — publish that here.
    """
    nested = msg.get("status")
    if not isinstance(nested, dict):
        return
    status = dict(nested)
    status["type"] = "status"
    status.setdefault("loaded", True)
    hub.publish(status)


def lab_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _no_window_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))


class WorkerBridge:
    """Host side: listen, spawn worker, forward commands, publish packets."""

    def __init__(self) -> None:
        self.port = 0
        self._sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._ready = threading.Event()
        self._queued: list[dict[str, Any]] = []
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._jpeg_inflight: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._proc: subprocess.Popen[Any] | None = None
        self._respawned = False
        self._closed = False
        self.op_wait = {"start": 120.0, "track": 120.0, "stop": 8.0, "calibrate": 8.0}

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def listen(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, 0))
        sock.listen(1)
        self._sock = sock
        self.port = int(sock.getsockname()[1])
        threading.Thread(target=self._accept_loop, daemon=True, name="lab-ipc-accept").start()
        return self.port

    def spawn(self) -> subprocess.Popen[Any] | None:
        if os.environ.get(NO_WORKER) == "1":
            _log("TRACK_LAB_NO_WORKER=1 — not spawning tracker")
            return None
        if self.port <= 0:
            raise RuntimeError("listen() before spawn()")
        root = lab_root()
        env = os.environ.copy()
        env[ENV_PORT] = str(self.port)
        existing = str(env.get("PYTHONPATH") or "")
        env["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")
        _log(f"starting worker python -m backend.worker ipc={self.port}")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "backend.worker"],
            cwd=str(root),
            env=env,
            stdout=None,
            stderr=None,
            creationflags=_no_window_flags(),
        )
        return self._proc

    def install(self) -> None:
        """Warming packets + command handler. Call before uvicorn serves."""
        publish_warming()
        hub.set_handler(self.handle_command)

    def handle_command(self, msg: dict[str, Any]) -> dict[str, Any]:
        ident = msg.get("id")
        if ident is None:
            ident = uuid.uuid4().hex
            msg = dict(msg)
            msg["id"] = ident
        op = str(msg.get("op") or "")
        if not self._ready.is_set():
            if op in LIVE_OPS:
                wait = float(self.op_wait.get(op, 8.0))
                if not self._ready.wait(wait):
                    _log(f"op={op} tracker still loading")
                    return ack(
                        ident=ident,
                        ok=False,
                        error="Track Lab tracker is still loading — wait a moment",
                        status=hub.latest_status() or warming_status(),
                    )
                return self._rpc(msg)
            with self._lock:
                if not self._ready.is_set():
                    self._queued.append(dict(msg))
                    _log(f"queued op={msg.get('op')} until tracker loads")
                    return ack(
                        ident=ident,
                        ok=True,
                        status=hub.latest_status() or warming_status(),
                    )
        return self._rpc(msg)

    def jpeg(self, kind: str) -> bytes | None:
        if not self._ready.is_set() or self._conn is None:
            return None
        kind = str(kind)
        ident = ""
        with self._lock:
            inflight = self._jpeg_inflight.get(kind)
            if inflight is None:
                ident = uuid.uuid4().hex
                ev = threading.Event()
                slot: dict[str, Any] = {}
                self._pending[ident] = (ev, slot)
                self._jpeg_inflight[kind] = (ev, slot)
                conn = self._conn
                start = True
            else:
                ev, slot = inflight
                conn = self._conn
                start = False
        if start:
            try:
                send(conn, {"type": "jpeg", "id": ident, "kind": kind}, self._send_lock)
            except OSError:
                with self._lock:
                    self._pending.pop(ident, None)
                    current = self._jpeg_inflight.get(kind)
                    if current is not None and current[0] is ev:
                        self._jpeg_inflight.pop(kind, None)
                ev.set()
                return None
        if not ev.wait(JPEG_WAIT):
            with self._lock:
                if ident:
                    self._pending.pop(ident, None)
                current = self._jpeg_inflight.get(kind)
                if current is not None and current[0] is ev:
                    self._jpeg_inflight.pop(kind, None)
            return None
        raw = slot.get("data")
        return raw if isinstance(raw, bytes) else None

    def close(self) -> None:
        self._closed = True
        self._ready.clear()
        conn = self._conn
        listen = self._sock
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if listen is not None:
            try:
                listen.close()
            except OSError:
                pass
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        ident = str(msg.get("id") or uuid.uuid4().hex)
        msg = dict(msg)
        msg["id"] = ident
        ev = threading.Event()
        slot: dict[str, Any] = {}
        with self._lock:
            conn = self._conn
            self._pending[ident] = (ev, slot)
        if conn is None:
            return ack(ident=ident, ok=False, error="tracker disconnected")
        with self._cmd_lock:
            try:
                send(conn, {"type": "command", "id": ident, "op": msg.get("op"), "body": msg.get("body") or {}}, self._send_lock)
            except OSError as exc:
                with self._lock:
                    self._pending.pop(ident, None)
                return ack(ident=ident, ok=False, error=str(exc))
            wait = GEN_WAIT if str(msg.get("op") or "") == "generate" else CMD_WAIT
            if not ev.wait(wait):
                with self._lock:
                    self._pending.pop(ident, None)
                return ack(ident=ident, ok=False, error="tracker timed out")
        reply = slot.get("ack")
        if isinstance(reply, dict):
            return reply
        return ack(ident=ident, ok=False, error="empty tracker reply")

    def _accept_loop(self) -> None:
        listen = self._sock
        if listen is None:
            return
        while not self._closed:
            try:
                conn, _addr = listen.accept()
            except OSError:
                return
            _log("worker connected")
            with self._lock:
                old = self._conn
                self._conn = conn
            if old is not None and old is not conn:
                try:
                    old.close()
                except OSError:
                    pass
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._read_loop(conn)
            finally:
                with self._lock:
                    if self._conn is conn:
                        self._conn = None
                    self._ready.clear()
                try:
                    conn.close()
                except OSError:
                    pass
                if not self._closed:
                    _log("worker dropped — dummy packets")
                    publish_warming()
                    self._maybe_respawn()

    def _read_loop(self, conn: socket.socket) -> None:
        for msg in iter_messages(conn):
            kind = str(msg.get("type") or "")
            if kind == "hello":
                self._on_hello(conn)
                continue
            if kind == "packet":
                packet = msg.get("packet")
                if isinstance(packet, dict):
                    packet.setdefault("loaded", True)
                    hub.publish(packet)
                continue
            if kind == "ack":
                _publish_ack_status(msg)
                self._resolve(str(msg.get("id") or ""), {"ack": msg})
                continue
            if kind == "jpeg_data":
                raw = msg.get("data")
                data = None
                if isinstance(raw, str) and raw:
                    import base64

                    try:
                        data = base64.b64decode(raw)
                    except Exception:
                        data = None
                self._resolve(str(msg.get("id") or ""), {"data": data})

    def _on_hello(self, conn: socket.socket) -> None:
        with self._lock:
            queued = list(self._queued)
            self._queued.clear()
            self._ready.set()
        _log(f"tracker ready queued={len(queued)}")
        for item in queued:
            try:
                send(
                    conn,
                    {
                        "type": "command",
                        "id": item.get("id") or uuid.uuid4().hex,
                        "op": item.get("op"),
                        "body": item.get("body") or {},
                    },
                    self._send_lock,
                )
            except OSError as exc:
                _log(f"drain failed: {exc}")
                break

    def _resolve(self, ident: str, payload: dict[str, Any]) -> None:
        with self._lock:
            item = self._pending.pop(ident, None)
        if item is None:
            return
        ev, slot = item
        slot.update(payload)
        ev.set()
        with self._lock:
            for kind, pair in list(self._jpeg_inflight.items()):
                if pair[0] is ev:
                    self._jpeg_inflight.pop(kind, None)

    def _maybe_respawn(self) -> None:
        if self._respawned or self._closed:
            return
        self._respawned = True
        try:
            self.spawn()
        except Exception as exc:
            _log(f"respawn failed: {exc}")


bridge = WorkerBridge()


def install_host() -> WorkerBridge:
    """Warming harness + IPC listen. Does not spawn the worker."""
    if bridge.port <= 0:
        bridge.listen()
        os.environ[ENV_PORT] = str(bridge.port)
    bridge.install()
    _log(f"host IPC listen 127.0.0.1:{bridge.port}")
    return bridge


def spawn_tracker_worker() -> None:
    if os.environ.get(NO_WORKER) == "1":
        return
    if bridge.port <= 0:
        install_host()
    bridge.spawn()
