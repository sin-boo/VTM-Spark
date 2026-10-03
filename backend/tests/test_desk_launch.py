"""Desk launch must not hand off to a JSON stub on :8765."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading

from backend import __main__ as main
from backend.single_instance import probe_existing_api


def _serve(body: bytes) -> tuple[HTTPServer, int]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a) -> None:
            return

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, int(httpd.server_address[1])


def _linger() -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    return stop, thread


def test_probe_rejects_empty_json_health() -> None:
    httpd, port = _serve(b"{}")
    try:
        assert probe_existing_api("127.0.0.1", port) is False
    finally:
        httpd.shutdown()


def test_probe_accepts_desk_health() -> None:
    httpd, port = _serve(b'{"ok":"1"}')
    try:
        assert probe_existing_api("127.0.0.1", port) is True
    finally:
        httpd.shutdown()


def test_wait_for_server_ignores_tcp_without_health() -> None:
    httpd, port = _serve(b"{}")
    stop, thread = _linger()
    try:
        assert main._wait_for_server("127.0.0.1", port, timeout=0.9, thread=thread) is False
    finally:
        stop.set()
        httpd.shutdown()


def test_wait_for_server_accepts_health_ok() -> None:
    httpd, port = _serve(b'{"ok":"1"}')
    stop, thread = _linger()
    try:
        assert main._wait_for_server("127.0.0.1", port, timeout=2.0, thread=thread) is True
    finally:
        stop.set()
        httpd.shutdown()


def test_blocked_by_existing_reports_foreign_listener() -> None:
    httpd, port = _serve(b"{}")
    try:
        msg = main._blocked_by_existing("127.0.0.1", port)
        assert msg is not None
        assert "mock_boot" in msg
        assert "already running" not in msg
    finally:
        httpd.shutdown()


def test_blocked_by_existing_reports_live_desk() -> None:
    httpd, port = _serve(b'{"ok":"1"}')
    try:
        msg = main._blocked_by_existing("127.0.0.1", port)
        assert msg is not None
        assert "already running" in msg
    finally:
        httpd.shutdown()


def test_fast_kill_clears_mock_boot() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "packaging" / "kill-orphans.ps1"
    ).read_text(encoding="utf-8")
    fast_at = text.index("if ($Fast)")
    rest = text[fast_at:]
    fast = rest[: rest.index("if (-not $Fast)")]
    assert "_mock_boot" in fast
    assert "LocalPort 8765" in fast
    assert r"-m\s+backend" in fast


def test_thread_caps_leave_cores_for_a_game() -> None:
    def pools(cpus: int | None) -> set[str]:
        env = main.desk_thread_env(cpus)
        return {v for k, v in env.items() if k.endswith("_NUM_THREADS")}

    assert pools(4) == {"2"}
    assert pools(12) == {"4"}
    assert pools(32) == {"4"}
    assert pools(1) == {"1"}
    assert pools(None) == {"2"}
    env = main.desk_thread_env(8)
    assert {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"} <= set(env)
    assert env["KMP_BLOCKTIME"] == "0"
    assert [main.opencv_threads(n) for n in (1, 4, 8, 32, None)] == [1, 1, 2, 2, 1]
