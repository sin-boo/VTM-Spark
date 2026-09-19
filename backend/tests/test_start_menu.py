from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _start_menu_text() -> str:
    return (Path(__file__).resolve().parents[1] / "packaging" / "start-menu.ps1").read_text(
        encoding="utf-8"
    )


def _invoke_start_app_body() -> str:
    text = _start_menu_text()
    start = text.index("function Invoke-StartApp")
    end = text.index("Enable-PrettyConsole", start)
    return text[start:end]


def test_start_menu_waits_for_desk_window() -> None:
    text = _start_menu_text()
    assert "function Wait-VtmDeskWindow" in text
    assert "function Test-VtmDeskWindow" in text
    assert "MainWindowTitle" in text
    assert "FindWindowW" in text


def test_start_builds_ui_from_process_exit_code() -> None:
    text = _start_menu_text()
    start = text.index("function Invoke-BuildDeskUi")
    end = text.index("function Test-VtmDeskWindow", start)
    body = text[start:end]
    assert "Invoke-ProcessWithHeartbeat" in body
    assert "npm" in body
    assert "run" in body
    assert "build" in body
    assert "LASTEXITCODE" not in body


def test_start_skips_ui_build_when_source_is_unchanged() -> None:
    text = _start_menu_text()
    stale = text[text.index("function Test-UiStale") : text.index("function Invoke-BuildDeskUi")]
    assert '$_.Name -notlike "_*"' in stale
    assert "return $true" in stale
    body = _invoke_start_app_body()
    assert body.count("Invoke-BuildDeskUi") == 1
    assert body.index("Test-UiStale") < body.index("Start-Process")
    assert body.index("Invoke-BuildDeskUi") < body.index("Start-Process")


def test_start_app_keeps_console_until_splash() -> None:
    body = _invoke_start_app_body()
    assert "Start-Process" in body
    assert "Wait-VtmDeskWindow" in body
    assert "Show-StartFailure" in body
    assert body.index("Start-Process") < body.index("Wait-VtmDeskWindow")
    assert body.index("Wait-VtmDeskWindow") < body.index("Hide-VtmConsole")
    assert "This window stays until the splash appears." in body
    hide_before_start = body[: body.index("Start-Process")].count("Hide-VtmConsole")
    assert hide_before_start == 0


def test_start_repairs_dead_venv_home() -> None:
    text = _start_menu_text()
    assert "venv-home.ps1" in text
    assert "Repair-VtmVenvHome" in text
    assert "Test-VtmPythonExe" in text
    assert "Resolve-VtmDeskPython" in text
    body = _invoke_start_app_body()
    assert "Resolve-VtmDeskPython" in body
    assert body.index("Resolve-VtmDeskPython") < body.index("Start-Process")
    assert "Scripts\\pythonw.exe" not in body


def test_start_split_into_install_and_run() -> None:
    root = _root()
    pack = root / "backend" / "packaging"
    install = (root / "install.bat").read_text(encoding="utf-8")
    assert "-Action install" in install
    stub = (pack / "run-stub.cs").read_text(encoding="utf-8")
    assert "-Action run" in stub
    exe = root / "run.exe"
    assert exe.is_file()
    assert exe.read_bytes()[:2] == b"MZ"
    assert (pack / "run.ico").is_file()
    build_run = (pack / "build-run.ps1").read_text(encoding="utf-8")
    assert "backend.app_icon" in build_run
    assert not (root / "run.bat").exists()
    assert not (root / "start.bat").exists()
    assert not (root / "Run.lnk").exists()
    menu = _start_menu_text()
    assert '[ValidateSet("install", "run")]' in menu
    assert "function Show-Menu" not in menu
