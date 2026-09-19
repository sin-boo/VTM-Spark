"""Windows PowerShell 5.1 reads .ps1 as ANSI unless the file has a BOM."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_start_scripts_are_ascii() -> None:
    for name in ("start.ps1", "setup.ps1", "start.bat"):
        (ROOT / name).read_bytes().decode("ascii")


def test_start_ps1_parses_in_windows_powershell() -> None:
    script = ROOT / "start.ps1"
    cmd = (
        "$err = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script.as_posix()}', [ref]$null, [ref]$err); "
        "if ($err) { $err | ForEach-Object { $_.ToString() }; exit 1 }"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
