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


def _function_body(name: str) -> str:
    text = _start_menu_text()
    start = text.index(f"function {name}")
    end = text.index("\nfunction ", start + 1)
    return text[start:end]


def test_install_bat_returns_to_caller_with_exit_code() -> None:
    install = (_root() / "install.bat").read_text(encoding="utf-8")
    assert "exit /b %EC%" in install
    lines = [line.strip().lower() for line in install.splitlines()]
    assert not any(line.endswith("exit %ec%") and "/b" not in line for line in lines)


def test_install_exits_with_script_exit_code() -> None:
    text = _start_menu_text()
    tail = text[text.index('if ($Action -eq "install")') :]
    assert "exit $script:InstallExitCode" in tail
    body = _function_body("Invoke-SmartBuild")
    assert "$script:InstallExitCode = 0" in body
    assert "$script:InstallExitCode = 1" in body
    assert "$script:InstallExitCode = 2" in body


def test_install_summary_uses_step_results() -> None:
    body = _function_body("Invoke-SmartBuild")
    assert "[void](Invoke-EnsureModel)" not in body
    assert "[void](Invoke-EnsureVtmNobleCam)" not in body
    assert "$modelsOk" in body
    assert "$vcamOk" in body
    assert "$trackLabOk" in body
    assert "Rebuild finished." not in body
    ok = body.index('"Install finished." green')
    assert body.rindex("if ($allOk)", 0, ok) < ok
    assert "some steps need attention" in body
    for label in ("App build", "Models", "Virtual camera", "Track Lab"):
        assert f'Write-InstallStep "{label}"' in body


def test_install_build_failure_has_hints() -> None:
    body = _function_body("Invoke-SmartBuild")
    assert "Rebuild failed" not in body
    for hint in ("internet", "Node.js", "NVIDIA driver", "Antivirus", ".venv-build"):
        assert hint in body
    assert "Re-running install.bat is safe" in body


def test_install_sets_up_track_lab_after_build() -> None:
    body = _function_body("Invoke-SmartBuild")
    assert body.index("$BuildScript") < body.index("Invoke-EnsureTrackLab")
    lab = _function_body("Invoke-EnsureTrackLab")
    assert "setup.ps1" in lab
    assert "Invoke-ProcessWithHeartbeat" in lab
    assert '"npm", "ci"' in lab
    assert '"npm", "install"' in lab
    assert "node_modules" in lab
    assert "LASTEXITCODE" not in lab


def test_model_step_keeps_python_stdout_out_of_return_value() -> None:
    body = _function_body("Invoke-EnsureModel")
    assert "& $VenvPy -m backend.model_download | Out-Host" in body
    assert "& $VenvPy -m backend.model_download --checklist-only | Out-Host" in body
    assert "Rebuild" not in body


def test_track_lab_scripts_point_at_install_bat() -> None:
    lab = _root() / "track_lab"
    for name in ("setup.ps1", "start.ps1", "README.md"):
        text = (lab / name).read_text(encoding="utf-8")
        assert "Smart Build" not in text, name
        assert "start.bat ->" not in text, name
    assert "install.bat" in (lab / "setup.ps1").read_text(encoding="utf-8")
    assert "install.bat" in (lab / "start.ps1").read_text(encoding="utf-8")


def test_edited_ps1_files_parse() -> None:
    import subprocess

    root = _root()
    for rel in ("backend/packaging/start-menu.ps1", "track_lab/setup.ps1", "track_lab/start.ps1"):
        script = (root / rel).as_posix()
        cmd = (
            "$err = $null; "
            "[void][System.Management.Automation.Language.Parser]::ParseFile("
            f"'{script}', [ref]$null, [ref]$err); "
            "if ($err) { $err | ForEach-Object { $_.ToString() }; exit 1 }"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, rel + ": " + out.stdout + out.stderr


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
