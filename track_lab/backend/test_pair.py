"""API and UI as one: close either and both stop."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
    )
    return str(pid) in out.stdout


@pytest.mark.skipif(os.name != "nt", reason="job objects are Windows only")
def test_killing_the_api_process_kills_the_ui(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "from backend.pair import _bind_to_job\n"
        "_bind_to_job()\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pid_file}', 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", script], cwd=str(ROOT))
    try:
        deadline = time.monotonic() + 20
        while not pid_file.exists() or not pid_file.read_text():
            assert time.monotonic() < deadline, "child never started"
            time.sleep(0.1)
        child = int(pid_file.read_text())
        assert _alive(child)
        parent.kill()  # like closing the window: no cleanup code runs
        parent.wait(timeout=10)
        deadline = time.monotonic() + 10
        while _alive(child):
            assert time.monotonic() < deadline, "UI outlived the API process"
            time.sleep(0.1)
    finally:
        if parent.poll() is None:
            parent.kill()


def test_ui_exit_stops_the_api(monkeypatch) -> None:
    from backend import pair

    class FakeServer:
        def __init__(self, config: object) -> None:
            self.started = False
            self.should_exit = False

        def run(self) -> None:
            self.started = True
            deadline = time.monotonic() + 10
            while not self.should_exit:
                assert time.monotonic() < deadline, "API kept running after the UI quit"
                time.sleep(0.05)

    stopped: list[object] = []
    monkeypatch.setattr(pair, "_bind_to_job", lambda: None)
    monkeypatch.setattr(pair, "probe_state", lambda *_a: "free")
    monkeypatch.setattr(pair, "claim_port", lambda: None)
    monkeypatch.setattr(
        pair, "_start_ui", lambda: subprocess.Popen([sys.executable, "-c", "pass"])
    )
    monkeypatch.setattr(pair, "_stop_ui", lambda ui: stopped.append(ui))
    import uvicorn

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    assert pair.main() == 0
    assert stopped and stopped[0] is not None


def test_find_npm_prefers_portable_node(tmp_path: Path, monkeypatch) -> None:
    from backend import pair

    ui = tmp_path / "track_lab" / "ui"
    ui.mkdir(parents=True)
    monkeypatch.setattr(pair, "UI_DIR", ui)
    monkeypatch.setattr(pair.shutil, "which", lambda _name: r"C:\system\npm.cmd")
    assert pair._find_npm() == r"C:\system\npm.cmd"

    portable = tmp_path / ".tools" / "node" / "npm.cmd"
    portable.parent.mkdir(parents=True)
    portable.write_text("", encoding="utf-8")
    assert pair._find_npm() == str(portable)
