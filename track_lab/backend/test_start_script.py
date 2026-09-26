"""Windows PowerShell 5.1 reads .ps1 as ANSI unless the file has a BOM."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_start_scripts_are_ascii() -> None:
    for name in ("start.ps1", "setup.ps1", "start.bat"):
        (ROOT / name).read_bytes().decode("ascii")


def test_start_ps1_uses_shared_osf_models() -> None:
    text = (ROOT / "start.ps1").read_text(encoding="ascii")
    assert r"vendor\tools\openseeface\models\lm_model3_opt.onnx" in text
    assert r'Join-Path $Root "models\lm_model3_opt.onnx"' not in text


def test_lab_weights_point_at_desk_trees() -> None:
    from backend.anime import ANIME_DIR
    from backend.osf_cam import MODELS_DIR
    from backend.paths import OSF_MODELS, TRACKERS

    assert MODELS_DIR == OSF_MODELS
    assert "openseeface" in MODELS_DIR.as_posix()
    assert ANIME_DIR == TRACKERS
    assert ANIME_DIR.as_posix().endswith("models/trackers")


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
