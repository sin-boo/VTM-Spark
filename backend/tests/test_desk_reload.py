from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from backend import desk_reload


def _hold():
    path = Path(__file__).resolve().parents[1] / "packaging" / "reload_hold.py"
    spec = importlib.util.spec_from_file_location("vtm_reload_hold", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def test_hold_script_is_outside_backend_package() -> None:
    source = desk_reload.hold_script().read_text(encoding="utf-8")
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(imports)
    assert "backend" not in joined
    assert "torch" not in joined
    assert "webview" not in joined
    assert "uvicorn" not in joined
    assert "never import backend" in source


def test_current_flags_drop_module_invocation() -> None:
    assert desk_reload.current_flags(["-m", "backend", "--ui", "dev"]) == ["--ui", "dev"]
    assert desk_reload.current_flags(["--port", "8765"]) == ["--port", "8765"]


def test_hold_command_points_at_standalone_script() -> None:
    cmd = desk_reload.hold_command(
        python=r"C:\py\python.exe",
        pid=4242,
        host="127.0.0.1",
        port=8765,
        cwd=r"F:\desk",
        flags=["--ui", "dev"],
    )
    assert cmd[0] == r"C:\py\python.exe"
    assert cmd[1] == str(desk_reload.hold_script())
    assert cmd[cmd.index("--pid") + 1] == "4242"
    assert cmd[cmd.index("--python") + 1] == r"C:\py\python.exe"
    assert cmd[-3:] == ["--", "--ui", "dev"]


def test_spawn_hold_is_detached_from_the_desk(monkeypatch) -> None:
    seen: dict = {}

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return SimpleNamespace(pid=9)

    monkeypatch.setattr(desk_reload.subprocess, "Popen", fake_popen)
    cmd = desk_reload.spawn_hold(
        python="python",
        pid=11,
        cwd=".",
        flags=[],
        popen=fake_popen,
    )
    assert Path(cmd[1]).name == "reload_hold.py"
    flags = seen["kw"].get("creationflags", 0)
    if flags:
        assert flags & desk_reload.DETACHED_PROCESS
        assert flags & desk_reload.CREATE_BREAKAWAY_FROM_JOB
        assert flags & desk_reload.CREATE_NEW_PROCESS_GROUP


def test_request_reload_opens_hold_then_exits(monkeypatch) -> None:
    desk_reload._requested = False
    calls: list[str] = []

    def fake_spawn(**_kw):
        calls.append("spawn")
        return ["hold"]

    monkeypatch.setattr(desk_reload.threading, "Timer", lambda *_a, **_k: SimpleNamespace(start=lambda: calls.append("exit")))
    out = desk_reload.request_reload(apply=True, spawn=fake_spawn, exit_fn=lambda: calls.append("exit-fn"))
    assert out["ok"] is True
    assert out["reloading"] is True
    assert calls[0] == "spawn"
    assert "exit" in calls
    desk_reload._requested = False


def test_hold_parse_and_backend_command() -> None:
    hold, _path = _hold()
    cfg = hold.parse_args(
        [
            "--pid",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--cwd",
            ".",
            "--python",
            "python",
            "--",
            "--ui",
            "webview",
        ]
    )
    assert cfg.flags == ("--ui", "webview")
    assert hold.backend_command(cfg.python, cfg.flags) == [
        "python",
        "-m",
        "backend",
        "--ui",
        "webview",
    ]


def test_hold_run_reload_waits_for_health(monkeypatch) -> None:
    hold, _path = _hold()
    cfg = hold.HoldConfig(
        pid=0,
        host="127.0.0.1",
        port=8765,
        cwd=".",
        python="python",
        flags=(),
        wait_dead=0.2,
        wait_up=0.6,
    )
    monkeypatch.setattr(hold, "health_ok", lambda *_a, **_k: True)
    seen = {}

    def fake_spawn(c):
        seen["cfg"] = c
        return SimpleNamespace(pid=1)

    assert hold.run_reload(cfg, spawn=fake_spawn) == 0
    assert seen["cfg"] is cfg


def test_api_reload_route_uses_hold_window() -> None:
    text = Path(__file__).resolve().parents[1] / "api.py"
    src = text.read_text(encoding="utf-8")
    assert '@app.post("/api/reload")' in src
    assert "request_reload" in src
    assert "desk_reload" in src


def test_settings_has_reload_backend_button() -> None:
    ui = Path(__file__).resolve().parents[2] / "ui" / "src"
    rail = (ui / "components" / "ControlRail.tsx").read_text(encoding="utf-8")
    app = (ui / "App.tsx").read_text(encoding="utf-8")
    api = (ui / "api.ts").read_text(encoding="utf-8")
    assert "Reload backend" in rail
    assert "onReloadBackend" in rail
    assert "api.reloadBackend" in app
    assert "fetch('/api/reload'" in api
