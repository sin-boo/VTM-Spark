import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _build() -> Path:
    return Path(__file__).resolve().parents[1] / "packaging" / "build.ps1"


def _text() -> str:
    return _build().read_text(encoding="utf-8-sig")


@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="requires Windows PowerShell",
)
def test_build_script_parses() -> None:
    script = (
        "$t = $null; $e = $null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{_build()}', [ref]$t, [ref]$e); "
        "if ($e.Count) { $e | ForEach-Object { $_.ToString() }; exit 1 }"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_uv_download_is_pinned() -> None:
    text = _text()
    assert "releases/latest" not in text
    assert re.search(r'\$UvVersion\s*=\s*"\d+\.\d+\.\d+"', text)
    assert "releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" in text
    # Unique temp paths, cleaned up afterwards.
    assert "NewGuid" in text
    assert "Remove-Item -LiteralPath $zip" in text


def test_missing_python_falls_back_to_uv_managed_python() -> None:
    text = _text()
    assert "No Python 3.10+ found. Pass -BasePython" not in text
    assert re.search(r'\$ManagedPythonVersion\s*=\s*"3\.1[0-3]"', text)
    assert '"--python", $VenvPython' in text


def _node_tools() -> Path:
    return _build().parent / "node-tools.ps1"


def test_portable_node_is_set_up_before_ui_build() -> None:
    text = _text()
    assert '. (Join-Path $PSScriptRoot "node-tools.ps1")' in text
    setup = text.index("Use-VtmNode -RepoRoot $Root -Install")
    ui = text.index('Write-Host "==> Building UI (Vite)"')
    assert setup < ui
    # Users no longer install Node themselves.
    assert "Install Node.js LTS from https://nodejs.org/" not in text


def test_portable_node_download_is_pinned_and_verified() -> None:
    text = _node_tools().read_text(encoding="utf-8")
    assert text.isascii()  # PS 5.1 reads BOM-less scripts as ANSI
    assert re.search(r'\$NodeVersion\s*=\s*"\d+\.\d+\.\d+"', text)
    assert "https://nodejs.org/dist/v$NodeVersion/$NodeZipName" in text
    assert re.search(r'\$NodeZipSha256\s*=\s*"[0-9a-f]{64}"', text)
    assert "Get-FileHash" in text
    assert "NewGuid" in text
    assert "Remove-Item -LiteralPath $zip" in text


@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="requires Windows PowerShell",
)
def test_node_tools_script_parses() -> None:
    script = (
        "$t = $null; $e = $null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{_node_tools()}', [ref]$t, [ref]$e); "
        "if ($e.Count) { $e | ForEach-Object { $_.ToString() }; exit 1 }"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_missing_nvidia_driver_only_warns() -> None:
    text = _text()
    assert "nvidia-smi" in text
    assert "WARNING: VTM Spark needs an NVIDIA GPU" in text
    nvidia = text.index("Test-NvidiaDriver")
    ui = text.index('Write-Host "==> Building UI (Vite)"')
    assert nvidia < ui


def test_throws_say_what_to_do_next() -> None:
    for msg in re.findall(r'throw "([^"]*)"', _text()):
        assert re.search(r"re-run|install\.bat|-BasePython|PATH|delete", msg, re.I), msg


def test_build_retries_instead_of_asking_the_user() -> None:
    text = _text()
    # npm: a broken node_modules is cleared and reinstalled automatically.
    assert "function Reset-UiPackages" in text
    assert text.count("Reset-UiPackages") >= 3
    # venv: a failed create is cleared and retried on uv's managed Python.
    assert '"uv venv (retry)"' in text
    # No "go install X yourself" steps.
    assert "python.org" not in text
    assert r"delete ui\node_modules" not in text
