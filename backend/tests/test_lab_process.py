from __future__ import annotations

import json

from backend import lab_process


def test_harness_ok_reads_body_larger_than_4k(monkeypatch) -> None:
    payload = {
        "type": "status",
        "protocol": "track_lab.harness.v1",
        "loaded": True,
        "pad": "x" * 5000,
    }
    raw = json.dumps(payload).encode("utf-8")
    assert len(raw) > 4096
    reads: list[int | None] = []

    class _Resp:
        status = 200

        def read(self, n: int | None = -1) -> bytes:
            reads.append(n)
            if n is None or n < 0:
                return raw
            return raw[:n]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(lab_process.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(lab_process, "_probe_fail", "down")
    assert lab_process.harness_ok() is True
    assert reads == [-1]


def test_ensure_lab_reuses_live_harness(monkeypatch) -> None:
    monkeypatch.setattr(lab_process, "harness_ok", lambda timeout=0.6: True)
    started: list[int] = []
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: started.append(-1) or [])
    monkeypatch.setattr(lab_process, "_start_lab", lambda: started.append(1))
    assert lab_process.ensure_lab() is True
    assert started == []


def test_ensure_lab_starts_after_stale(monkeypatch) -> None:
    live = {"ok": False}

    def fake_ok(timeout: float = 0.6) -> bool:
        return live["ok"]

    class _Proc:
        def poll(self) -> None:
            return None

    def fake_start():
        live["ok"] = True
        return _Proc()

    monkeypatch.setattr(lab_process, "harness_ok", fake_ok)
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: [1])
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [])
    monkeypatch.setattr(lab_process, "_start_lab", fake_start)
    monkeypatch.setattr(lab_process, "_owned", None)
    assert lab_process.ensure_lab(timeout=2.0) is True


def test_connect_lab_handshakes_without_start(monkeypatch) -> None:
    monkeypatch.setattr(lab_process, "ensure_lab", lambda timeout=12.0, on_wait=None: True)

    class _Lab:
        def handshake(self) -> dict:
            return {
                "online": True,
                "ok": True,
                "handshake": True,
                "live": False,
                "error": "",
            }

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    packet = lab_process.connect_lab()
    assert packet["handshake"] is True
    assert packet["online"] is True
    assert packet["live"] is False


def test_connect_lab_reports_offline_when_lab_missing(monkeypatch) -> None:
    monkeypatch.setattr(lab_process, "ensure_lab", lambda timeout=12.0, on_wait=None: False)
    packet = lab_process.connect_lab()
    assert packet["online"] is False
    assert packet["handshake"] is False
    assert "did not start" in packet["error"]


def test_spawn_lab_does_not_wait(monkeypatch) -> None:
    started: list[int] = []

    class _Proc:
        def poll(self) -> None:
            return None

    monkeypatch.setattr(lab_process, "harness_ok", lambda timeout=0.6: False)
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: [])
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [])
    monkeypatch.setattr(lab_process, "_start_lab", lambda: started.append(1) or _Proc())
    monkeypatch.setattr(lab_process, "_owned", None)
    assert lab_process.spawn_lab() is True
    assert started == [1]


def test_kill_stale_hides_taskkill(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(_cmd, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(lab_process, "stale_pids", lambda port=8780: [42])
    monkeypatch.setattr(lab_process.subprocess, "run", fake_run)
    lab_process.kill_stale()
    if lab_process.os.name == "nt":
        assert captured.get("creationflags") == lab_process.lab_creationflags()
    assert captured.get("stdout") is not None
    assert captured.get("stderr") is not None


def test_start_lab_hides_console(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    class _Proc:
        pass

    def fake_popen(*_args, **kwargs):
        captured.update(kwargs)
        return _Proc()

    py = tmp_path / "python.exe"
    py.write_bytes(b"x")
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(lab_process, "venv_python", lambda: py)
    monkeypatch.setattr(lab_process, "lab_root", lambda: tmp_path)
    monkeypatch.setattr(lab_process, "package_root", lambda: tmp_path)
    monkeypatch.setattr(lab_process, "_copy_osf", lambda: None)
    monkeypatch.setattr(lab_process.subprocess, "Popen", fake_popen)
    lab_process._start_lab()
    assert captured.get("creationflags") == lab_process.lab_creationflags()
    assert captured.get("stdout") is not None
    assert captured.get("stderr") is not None
    env = captured.get("env") or {}
    assert env.get("OPENCV_VIDEOIO_PRIORITY_MSMF") == "0"
    if lab_process.os.name == "nt":
        assert captured["creationflags"] == 0x08000000


def test_ensure_lab_reports_wait_progress(monkeypatch) -> None:
    hits = {"n": 0}
    seen: list[float] = []

    def fake_ok(timeout: float = 0.6) -> bool:
        hits["n"] += 1
        return hits["n"] >= 4

    class _Proc:
        def poll(self) -> None:
            return None

    monkeypatch.setattr(lab_process, "harness_ok", fake_ok)
    monkeypatch.setattr(lab_process, "spawn_lab", lambda: True)
    monkeypatch.setattr(lab_process, "_owned", _Proc())
    assert lab_process.ensure_lab(timeout=2.0, on_wait=seen.append) is True
    assert seen
    assert seen[-1] == 1.0


def test_connect_wait_is_short_for_ui_poll() -> None:
    assert lab_process.CONNECT_WAIT <= 5.0
    assert lab_process.BOOT_WAIT >= 60.0


def test_start_tracking_waits_for_lab_not_live_poser() -> None:
    import inspect

    from backend.stream import StreamRuntime

    src = inspect.getsource(StreamRuntime.start_tracking)
    assert "_wait_lab_for_tracking" in src
    assert "_start_legacy_tracking" not in src
    assert "Stop stream before starting tracking" not in src
    assert "_set_track_status" in src
    wait = inspect.getsource(StreamRuntime._wait_lab_for_tracking)
    assert "wait_loaded" in wait
    assert "tracker_loaded" in wait
    assert "harness_ok" in wait
    assert "BOOT_WAIT" in wait
    assert "CONNECT_WAIT" not in wait


def test_start_lab_tracking_fits_rest_before_live() -> None:
    import inspect
    from pathlib import Path

    from backend.stream import StreamRuntime

    src = inspect.getsource(StreamRuntime._start_lab_tracking)
    assert "_sync_lab_character" in src
    assert '_lab_ack("start"' in src
    assert src.index("_sync_lab_character") < src.index('_lab_ack("start"')
    boot = inspect.getsource(StreamRuntime._boot_lab_source)
    assert "_sync_lab_character" in boot
    assert "replace=True" not in boot
    sync = inspect.getsource(StreamRuntime._sync_lab_character)
    assert "put_source" in src or "put_source" in sync
    assert "lab_keeps_authored" in sync
    assert "_same_lab_still" in sync
    assert '_lab_ack("track")' in sync
    assert "_adopt_lab_hair" in sync
    assert sync.index("lab_keeps_authored") < sync.index("put_source")
    assert sync.index("_same_lab_still") < sync.index("put_source")
    assert sync.index("put_source") < sync.index('_lab_ack("track")')
    assert "_adopt_lab_hair" in src
    boot_char = inspect.getsource(StreamRuntime._boot_character)
    assert "replace_lab=False" in boot_char
    live = (Path(__file__).resolve().parents[2] / "track_lab" / "backend" / "face.py").read_text(
        encoding="utf-8"
    )
    start = live.split("def start_live(", 1)[1].split("def set_camera(", 1)[0]
    assert "self.track()" in start
    assert start.index("self.track()") < start.index("Track a face first so rest exists")


def test_boot_runs_lab_handshake_in_parallel() -> None:
    import inspect

    from backend.stream import StreamRuntime

    src = inspect.getsource(StreamRuntime._run_boot)
    assert "vtm-boot-lab" in src
    assert "_boot_lab" in src
    assert "_boot_lab_source" in src
    boot = inspect.getsource(StreamRuntime._boot_lab)
    assert "BOOT_WAIT" in boot
    assert "CONNECT_WAIT" not in boot


def test_spawn_skips_kill_when_port_free(monkeypatch) -> None:
    started: list[int] = []
    killed: list[int] = []

    class _Proc:
        def poll(self) -> None:
            return None

    monkeypatch.setattr(lab_process, "harness_ok", lambda timeout=0.6: False)
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: killed.append(1) or [])
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [])
    monkeypatch.setattr(lab_process, "_start_lab", lambda: started.append(1) or _Proc())
    monkeypatch.setattr(lab_process, "_owned", None)
    assert lab_process.spawn_lab() is True
    assert started == [1]
    assert killed == []


def test_spawn_reuses_booting_listener(monkeypatch) -> None:
    started: list[int] = []
    killed: list[int] = []

    monkeypatch.setattr(lab_process, "harness_ok", lambda timeout=0.6: False)
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: killed.append(1) or [])
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [77])
    monkeypatch.setattr(lab_process, "_harness_http_code", lambda timeout=1.2: None)
    monkeypatch.setattr(lab_process, "_start_lab", lambda: started.append(1))
    monkeypatch.setattr(lab_process, "_owned", None)
    assert lab_process.spawn_lab() is True
    assert started == []
    assert killed == []


def test_spawn_kills_only_http_404(monkeypatch) -> None:
    started: list[int] = []

    class _Proc:
        def poll(self) -> None:
            return None

    monkeypatch.setattr(lab_process, "harness_ok", lambda timeout=0.6: False)
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: [77])
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [77])
    monkeypatch.setattr(lab_process, "_harness_http_code", lambda timeout=1.2: 404)
    monkeypatch.setattr(lab_process, "_start_lab", lambda: started.append(1) or _Proc())
    monkeypatch.setattr(lab_process, "_owned", None)
    assert lab_process.spawn_lab() is True
    assert started == [1]


def test_ensure_lab_hooks_exited_duplicate(monkeypatch) -> None:
    live = {"ok": False}

    def fake_ok(timeout: float = 0.6) -> bool:
        return live["ok"]

    class _Dead:
        def poll(self) -> int:
            live["ok"] = True
            return 0

        @property
        def returncode(self) -> int:
            return 0

    monkeypatch.setattr(lab_process, "harness_ok", fake_ok)
    monkeypatch.setattr(lab_process, "spawn_lab", lambda: True)
    monkeypatch.setattr(lab_process, "_owned", _Dead())
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [9])
    assert lab_process.ensure_lab(timeout=2.0) is True


def test_ensure_lab_waits_for_unowned_listener(monkeypatch) -> None:
    hits = {"n": 0}
    seen: list[float] = []

    def fake_ok(timeout: float = 0.6) -> bool:
        hits["n"] += 1
        return hits["n"] >= 4

    monkeypatch.setattr(lab_process, "harness_ok", fake_ok)
    monkeypatch.setattr(lab_process, "spawn_lab", lambda: True)
    monkeypatch.setattr(lab_process, "_owned", None)
    monkeypatch.setattr(lab_process, "_listener_pids", lambda port=8780: [9])
    assert lab_process.ensure_lab(timeout=2.0, on_wait=seen.append) is True
    assert seen[-1] == 1.0


def test_watch_lab_returns_when_already_started(monkeypatch) -> None:
    started: list[int] = []
    monkeypatch.setattr(lab_process, "spawn_lab", lambda: started.append(1) or True)
    monkeypatch.setattr(lab_process, "_watch_started", True)
    lab_process.watch_lab()
    assert started == []


def test_stop_owned_lab_kills_stale_without_owned(monkeypatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(lab_process, "_owned", None)
    monkeypatch.setattr(lab_process, "kill_stale", lambda port=8780: killed.append(1) or [])
    lab_process.stop_owned_lab()
    assert killed == [1]


def test_spawn_stops_foreign_live_lab_once(monkeypatch) -> None:
    stops: list[int] = []
    monkeypatch.setattr(lab_process, "harness_ok", lambda timeout=0.6: True)
    monkeypatch.setattr(lab_process, "_owned_alive", lambda: False)
    monkeypatch.setattr(lab_process, "_stopped_foreign", False)
    monkeypatch.setattr(
        lab_process, "_stop_foreign_lab_capture", lambda: stops.append(1)
    )
    monkeypatch.setattr(lab_process, "_start_lab", lambda: None)
    assert lab_process.spawn_lab() is True
    assert lab_process.spawn_lab() is True
    assert stops == [1]


