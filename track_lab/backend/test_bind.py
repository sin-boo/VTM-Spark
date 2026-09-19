from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from backend.bind import (
    looks_like_harness,
    looks_like_health,
    occupied_message,
    probe_state,
    stale_message,
    _local_port,
)
from backend.ports import SERVICE


def test_health_payload_must_name_the_lab() -> None:
    assert looks_like_health({"ok": True, "service": SERVICE})
    assert not looks_like_health({"ok": True})
    assert not looks_like_health({"ok": True, "service": "vtm_noble"})
    assert not looks_like_health("ok")


def test_harness_payload_is_not_plain_health() -> None:
    assert looks_like_harness({"type": "status", "ok": True})
    assert looks_like_harness({"protocol": "track_lab.harness.v1", "ok": True})
    assert not looks_like_harness({"ok": True, "service": SERVICE})
    assert not looks_like_harness({"ok": True})


def test_occupied_message_names_8780() -> None:
    assert "8780" in occupied_message()
    assert "another app" in occupied_message()


def test_stale_message_names_harness() -> None:
    assert "/harness/status" in stale_message()
    assert "old build" in stale_message()


def test_local_port_parses_ipv4_and_ipv6() -> None:
    assert _local_port("127.0.0.1:8780") == 8780
    assert _local_port("[::1]:8780") == 8780
    assert _local_port("0.0.0.0:80") == 80
    assert _local_port("nope") is None


def _serve(routes: dict[str, dict]) -> tuple[HTTPServer, int]:
    class _App(BaseHTTPRequestHandler):
        table = routes

        def do_GET(self) -> None:  # noqa: N802
            payload = self.table.get(self.path)
            if payload is None:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), _App)
    port = int(server.server_address[1])
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_probe_state_lab_when_harness_json_exceeds_4k() -> None:
    health = {"ok": True, "service": SERVICE}
    harness = {
        "type": "status",
        "protocol": "track_lab.harness.v1",
        "ok": True,
        "pad": "x" * 5000,
    }
    raw = json.dumps(harness).encode("utf-8")
    assert len(raw) > 4096
    server, port = _serve({"/api/health": health, "/harness/status": harness})
    try:
        assert probe_state("127.0.0.1", port) == "lab"
    finally:
        server.shutdown()
        server.server_close()


def test_probe_state_lab_stale_busy_free() -> None:
    health = {"ok": True, "service": SERVICE}
    harness = {"type": "status", "protocol": "track_lab.harness.v1", "ok": True}
    server, port = _serve({"/api/health": health, "/harness/status": harness})
    try:
        assert probe_state("127.0.0.1", port) == "lab"
    finally:
        server.shutdown()
        server.server_close()

    server, port = _serve({"/api/health": health})
    try:
        assert probe_state("127.0.0.1", port) == "stale"
    finally:
        server.shutdown()
        server.server_close()

    server, port = _serve({"/api/health": {"ok": True, "service": "other"}})
    try:
        # Python on 8780 without a harness is an old lab, not a foreign app.
        assert probe_state("127.0.0.1", port) == "stale"
    finally:
        server.shutdown()
        server.server_close()

    assert probe_state("127.0.0.1", 1) == "free"


def test_probe_state_busy_when_foreign_listener(monkeypatch) -> None:
    monkeypatch.setattr("backend.bind.probe_harness", lambda *a, **k: False)
    monkeypatch.setattr("backend.bind.probe_health", lambda *a, **k: False)
    monkeypatch.setattr("backend.bind.port_in_use", lambda *a, **k: True)
    monkeypatch.setattr("backend.bind.python_listeners", lambda *a, **k: [])
    monkeypatch.setattr("backend.bind.harness_http_status", lambda *a, **k: None)
    assert probe_state("127.0.0.1", 8780) == "busy"


def test_probe_state_404_without_health_is_stale() -> None:
    server, port = _serve({})
    try:
        assert probe_state("127.0.0.1", port) == "stale"
    finally:
        server.shutdown()
        server.server_close()
