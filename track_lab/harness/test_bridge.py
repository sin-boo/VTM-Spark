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


def test_bridge_ack_publishes_input_source() -> None:
    """set_input's reply is the status the desk polls, even if no status packet arrived."""
    previous = hub._handler
    previous_status = hub._status
    previous_frame = hub._frame
    hub._status = {"type": "status", "source": "camera", "loaded": True}
    bridge = WorkerBridge()
    port = bridge.listen()
    bridge.install()

    def worker() -> None:
        sock = connect(port, timeout=2.0)
        send(sock, {"type": "hello"})
        send(
            sock,
            {
                "type": "packet",
                "packet": {"type": "frame", "protocol": PROTOCOL, "source": "camera", "live": False},
            },
        )
        for msg in iter_messages(sock):
            if msg.get("type") != "command":
                continue
            send(
                sock,
                {
                    "type": "ack",
                    "id": msg.get("id"),
                    "ok": True,
                    "error": "",
                    "status": {"type": "status", "source": "ifm", "loaded": True, "live": False, "ok": True},
                },
            )
            break

    thread = threading.Thread(target=worker, daemon=True)
    try:
        thread.start()
        assert bridge._ready.wait(3.0)
        reply = bridge.handle_command({"op": "set_input", "body": {"source": "ifm"}})
        assert reply["ok"] is True
        assert hub.latest_status() is not None
        assert hub.latest_status()["source"] == "ifm"
        thread.join(timeout=3.0)
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


def _hub_state():
    return hub._handler, hub._status, hub._frame


def _put_hub_state(state) -> None:
    hub.set_handler(state[0])
    hub._status = state[1]
    hub._frame = state[2]


def _wait_for(check, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return bool(check())


def test_bridge_refuses_set_point_while_loading() -> None:
    """A nudge is measured on a live pose; the loading tracker has none. Queued
    and replayed, it was measured against the rest pose instead."""
    state = _hub_state()
    bridge = WorkerBridge()
    try:
        hub.set_handler(bridge.handle_command)
        reply = bridge.handle_command({"op": "set_point", "id": "p1", "body": {"id": 21, "x": 1.0, "y": 2.0}})
        assert reply["ok"] is False
        assert "still loading" in str(reply.get("error") or "")
        assert bridge._queued == []
    finally:
        _put_hub_state(state)


def test_bridge_replays_the_queue_before_a_waiting_live_op(monkeypatch) -> None:
    """set_source queued while the tracker loads, set_rest waiting for it to be
    ready. set_rest used to reach the worker first, and the replayed
    set_source then cleared the rest it had just installed."""
    import harness.bridge as bridge_mod

    real_send = bridge_mod.send

    def slow_replay(sock, msg, lock=None):
        if msg.get("op") == "set_source":
            time.sleep(0.3)
        real_send(sock, msg, lock)

    monkeypatch.setattr(bridge_mod, "send", slow_replay)
    state = _hub_state()
    bridge = WorkerBridge()
    port = bridge.listen()
    bridge.install()
    monkeypatch.setattr(bridge, "spawn", lambda: None)
    ops: list[str] = []
    replies: list[dict] = []

    def worker() -> None:
        sock = connect(port, timeout=2.0)
        send(sock, {"type": "hello"})
        for msg in iter_messages(sock):
            if msg.get("type") != "command":
                continue
            ops.append(str(msg.get("op")))
            send(sock, {"type": "ack", "id": msg.get("id"), "ok": True, "error": "", "status": {"loaded": True}})
            if len(ops) == 2:
                break

    try:
        queued = bridge.handle_command({"op": "set_source", "id": "q1", "body": {"path": "x.png"}})
        assert queued["ok"] is True
        waiter = threading.Thread(
            target=lambda: replies.append(bridge.handle_command({"op": "set_rest", "id": "r1", "body": {}})),
            daemon=True,
        )
        waiter.start()
        time.sleep(0.05)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        waiter.join(timeout=5.0)
        thread.join(timeout=5.0)
        assert ops == ["set_source", "set_rest"]
        assert replies and replies[0]["ok"] is True
    finally:
        bridge.close()
        _put_hub_state(state)


def test_bridge_fails_a_command_in_flight_when_the_worker_drops(monkeypatch) -> None:
    """No reply comes from a dead worker. The command waited out CMD_WAIT (3
    minutes) holding the command lock, so nothing reached the new worker."""
    monkeypatch.setattr("harness.bridge.CMD_WAIT", 5.0)
    state = _hub_state()
    bridge = WorkerBridge()
    port = bridge.listen()
    bridge.install()
    monkeypatch.setattr(bridge, "spawn", lambda: None)

    def worker() -> None:
        sock = connect(port, timeout=2.0)
        send(sock, {"type": "hello"})
        for msg in iter_messages(sock):
            if msg.get("type") == "command":
                break
        sock.close()

    thread = threading.Thread(target=worker, daemon=True)
    try:
        thread.start()
        assert bridge._ready.wait(3.0)
        started = time.monotonic()
        reply = bridge.handle_command({"op": "set_feel", "body": {"mouth": 1.0}})
        assert reply["ok"] is False
        assert "disconnected" in str(reply.get("error") or "")
        assert time.monotonic() - started < 4.0
        assert bridge._pending == {}
    finally:
        bridge.close()
        _put_hub_state(state)


def _drop_workers(monkeypatch, *, healthy: float, times: int) -> list[bool]:
    monkeypatch.setattr("harness.bridge.RESPAWN_HEALTHY", healthy)
    state = _hub_state()
    bridge = WorkerBridge()
    port = bridge.listen()
    bridge.install()
    spawned: list[bool] = []
    monkeypatch.setattr(bridge, "spawn", lambda: spawned.append(True))
    try:
        for n in range(times):
            sock = connect(port, timeout=2.0)
            send(sock, {"type": "hello"})
            assert _wait_for(bridge._ready.is_set)
            sock.close()
            assert _wait_for(lambda: not bridge._ready.is_set())
            # The drop is handled once the next accept is listening again.
            _wait_for(lambda: len(spawned) > n, timeout=0.5)
    finally:
        bridge.close()
        _put_hub_state(state)
    return spawned


def test_bridge_respawns_again_after_a_healthy_run(monkeypatch) -> None:
    """Only the first drop was ever respawned: a second crash left the desk on
    warming packets until Track Lab was restarted by hand."""
    assert len(_drop_workers(monkeypatch, healthy=0.0, times=3)) == 3


def test_bridge_does_not_respawn_a_crash_loop(monkeypatch) -> None:
    assert len(_drop_workers(monkeypatch, healthy=3600.0, times=3)) == 1
