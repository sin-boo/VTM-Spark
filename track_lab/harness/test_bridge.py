from __future__ import annotations

import threading
import time

from harness.bridge import WorkerBridge, connect, iter_messages, send
from harness.hub import hub
from harness.protocol import PROTOCOL


def test_bridge_calibrate_fails_until_ready() -> None:
    previous = hub._handler
    bridge = WorkerBridge()
    try:
        hub.set_handler(bridge.handle_command)
        bridge.op_wait["calibrate"] = 0.05
        reply = bridge.handle_command({"op": "calibrate", "id": "c1", "body": {"id": "rest"}})
        assert reply["ok"] is False
        assert "still loading" in str(reply.get("error") or "")
        assert bridge._queued == []
    finally:
        hub.set_handler(previous)


def test_bridge_queues_commands_until_ready() -> None:
    previous = hub._handler
    bridge = WorkerBridge()
    try:
        hub.set_handler(bridge.handle_command)
        reply = bridge.handle_command({"op": "set_source", "id": "q1", "body": {"path": "x.png"}})
        assert reply["ok"] is True
        assert len(bridge._queued) == 1
        assert bridge._queued[0]["op"] == "set_source"
        ping = hub.command("ping")
        assert ping["ok"] is True
    finally:
        hub.set_handler(previous)


def test_bridge_fake_worker_loaded_and_rpc() -> None:
    previous = hub._handler
    previous_status = hub._status
    previous_frame = hub._frame
    bridge = WorkerBridge()
    port = bridge.listen()
    bridge.install()
    seen: list[dict] = []

    def worker() -> None:
        sock = connect(port, timeout=2.0)
        send(sock, {"type": "hello"})
        send(
            sock,
            {
                "type": "packet",
                "packet": {
                    "type": "status",
                    "protocol": PROTOCOL,
                    "loaded": True,
                    "ok": True,
                    "live": False,
                },
            },
        )
        for msg in iter_messages(sock):
            seen.append(msg)
            if msg.get("type") == "command":
                send(
                    sock,
                    {
                        "type": "ack",
                        "id": msg.get("id"),
                        "ok": True,
                        "error": "",
                        "status": {"loaded": True, "live": True, "ok": True},
                    },
                )
                break

    thread = threading.Thread(target=worker, daemon=True)
    try:
        thread.start()
        assert bridge._ready.wait(3.0)
        deadline = threading.Event()
        for _ in range(20):
            packet = hub.latest_status()
            if packet and packet.get("loaded"):
                break
            deadline.wait(0.05)
        assert hub.latest_status() is not None
        assert hub.latest_status().get("loaded") is True
        reply = bridge.handle_command({"op": "start", "body": {"camera": 0}})
        assert reply["ok"] is True
        thread.join(timeout=3.0)
        assert any(item.get("op") == "start" for item in seen)
    finally:
        bridge.close()
        hub.set_handler(previous)
        hub._status = previous_status
        hub._frame = previous_frame


def test_jpeg_coalesces_inflight_requests(monkeypatch) -> None:
    bridge = WorkerBridge()
    sent: list[dict] = []
    monkeypatch.setattr("harness.bridge.send", lambda sock, msg, lock=None: sent.append(msg))
    monkeypatch.setattr("harness.bridge.JPEG_WAIT", 1.0)
    class FakeConn:
        def close(self) -> None:
            return None

    bridge._ready.set()
    bridge._conn = FakeConn()
    results: list[bytes | None] = []

    def call() -> None:
        results.append(bridge.jpeg("camera"))

    first = threading.Thread(target=call)
    second = threading.Thread(target=call)
    first.start()
    deadline = time.perf_counter() + 0.5
    while not sent and time.perf_counter() < deadline:
        time.sleep(0.01)
    second.start()
    time.sleep(0.05)
    assert len(sent) == 1
    ident = str(sent[0].get("id") or "")
    bridge._resolve(ident, {"data": b"jpg"})
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert results == [b"jpg", b"jpg"]
    bridge.close()
