"""Torch / OSF worker. Host on 8780 is already answering harness HTTP.

Connects to TRACK_LAB_IPC_PORT after FaceBench loads, then forwards packed
frames and handles harness commands on this process (CUDA + DirectShow).
"""

from __future__ import annotations

import base64
import os
import threading

from harness.bridge import ENV_PORT, connect, iter_messages, send
from harness.dispatch import bind, handle
from harness.hub import LatestSlot, hub
from harness.pack import frame_from_bench, status_from_bench
from harness.protocol import ack


def _log(msg: str) -> None:
    print(f"[track-lab-worker] {msg}", flush=True)


def _wrap_publish() -> None:
    real = hub.publish
    pending = LatestSlot()

    def forward(packet: dict) -> None:
        payload = dict(packet)
        payload["loaded"] = True
        real(payload)
        pending.put({"type": "packet", "packet": payload})

    def sender() -> None:
        while True:
            msg = pending.take(timeout=0.25)
            if msg is None:
                continue
            sock = _sock
            if sock is None:
                continue
            try:
                send(sock, msg, _send_lock)
            except OSError:
                return

    hub.publish = forward  # type: ignore[method-assign]
    threading.Thread(target=sender, daemon=True, name="lab-ipc-send").start()


def _jpeg(bench, kind: str) -> bytes | None:
    if kind == "overlay":
        return bench.overlay_jpeg()
    if kind == "camera":
        return bench.camera_jpeg()
    if kind == "gen":
        from backend.vtm_gen import last_jpeg

        return last_jpeg()
    return bench.source_jpeg()


def _serve(bench, sock) -> None:
    bind(bench)
    send(sock, {"type": "hello"}, _send_lock)
    _log("hello — tracker attached")
    try:
        hub.publish(status_from_bench(bench, clients=hub.clients))
        hub.publish(frame_from_bench(bench, clients=hub.clients))
    except Exception as exc:
        _log(f"initial pack failed: {exc}")
    for msg in iter_messages(sock):
        kind = str(msg.get("type") or "")
        if kind == "jpeg":
            ident = msg.get("id")
            raw = None
            try:
                raw = _jpeg(bench, str(msg.get("kind") or "source"))
            except Exception as exc:
                _log(f"jpeg {msg.get('kind')} failed: {exc}")
            data = base64.b64encode(raw).decode("ascii") if raw else None
            send(sock, {"type": "jpeg_data", "id": ident, "data": data}, _send_lock)
            continue
        if kind != "command":
            continue
        ident = msg.get("id")
        try:
            reply = handle(bench, {"id": ident, "op": msg.get("op"), "body": msg.get("body") or {}})
        except Exception as exc:
            _log(f"op={msg.get('op')} raised: {exc}")
            reply = ack(ident=ident, ok=False, error=str(exc))
        out = dict(reply)
        out["type"] = "ack"
        out["id"] = ident
        send(sock, out, _send_lock)


_sock = None
_send_lock = threading.Lock()


def main() -> int:
    global _sock
    raw = os.environ.get(ENV_PORT) or ""
    try:
        port = int(raw)
    except ValueError:
        _log(f"missing {ENV_PORT}")
        return 2
    _log(f"loading FaceBench (torch) ipc={port}")
    from backend.face import bench

    _log("FaceBench ready — connecting to host")
    sock = connect(port)
    _sock = sock
    _wrap_publish()
    try:
        _serve(bench, sock)
    except (OSError, Exception) as exc:
        _log(f"ipc closed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
