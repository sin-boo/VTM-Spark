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


def test_npm_is_checked_before_ui_build() -> None:
    text = _text()
    check = text.index("Get-Command npm")
    ui = text.index('Write-Host "==> Building UI (Vite)"')
    assert check < ui
    assert "https://nodejs.org/" in text


def test_missing_nvidia_driver_only_warns() -> None:
    text = _text()
    assert "nvidia-smi" in text
    assert "WARNING: VTM Noble needs an NVIDIA GPU" in text
    nvidia = text.index("Test-NvidiaDriver")
    ui = text.index('Write-Host "==> Building UI (Vite)"')
    assert nvidia < ui


def test_throws_say_what_to_do_next() -> None:
    for msg in re.findall(r'throw "([^"]*)"', _text()):
        assert re.search(r"re-run|install\.bat|-BasePython|PATH|delete", msg, re.I), msg
