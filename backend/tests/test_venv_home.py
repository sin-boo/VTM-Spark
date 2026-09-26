import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_needs_powershell = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="requires Windows PowerShell",
)


def _pack() -> Path:
    return Path(__file__).resolve().parents[1] / "packaging"


def _ps_strict(script: str) -> subprocess.CompletedProcess:
    """Dot-source venv-home.ps1 under ErrorActionPreference=Stop, like build.ps1 does."""
    helper = _pack() / "venv-home.ps1"
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"$ErrorActionPreference = 'Stop'; . '{helper}'; {script}",
        ],
        capture_output=True,
        text=True,
    )


@_needs_powershell
def test_host_python_candidates_run_under_error_action_stop() -> None:
    # Regression: assigning $home (ReadOnly/AllScope $HOME) threw
    # "Cannot overwrite variable HOME" and killed every fresh install.
    completed = _ps_strict("Get-VtmHostPythonCandidates")
    assert completed.returncode == 0, completed.stderr
    assert "Cannot overwrite variable" not in (completed.stderr or "")
    lines = [ln.strip() for ln in (completed.stdout or "").splitlines() if ln.strip()]
    assert lines, "expected candidate paths"
    assert all(ln.lower().endswith("python.exe") for ln in lines)
    assert any("miniconda3" in ln.lower() for ln in lines)


@_needs_powershell
def test_resolve_matching_host_python_runs_under_error_action_stop() -> None:
    completed = _ps_strict("$null = Resolve-VtmMatchingHostPython -WantMinor '0.0'; 'ok'")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")


def test_venv_home_does_not_assign_automatic_variables() -> None:
    text = (_pack() / "venv-home.ps1").read_text(encoding="utf-8")
    auto = r"home|input|args|error|pid|host|matches|psitem|_|this|pwd|profile"
    assert not re.search(rf"(?im)^\s*\$({auto})\s*=(?!=)", text)


def _ps(script: str) -> str:
    helper = _pack() / "venv-home.ps1"
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f". '{helper}'; {script}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or "").strip()


def test_build_repairs_or_recreates_dead_venv() -> None:
    text = (_pack() / "build.ps1").read_text(encoding="utf-8")
    assert "venv-home.ps1" in text
    assert "Repair-VtmVenvHome" in text
    assert "Get-VtmHostPythonCandidates" in text
    assert "Recreating build venv" in text


def test_resolve_desk_python_skips_missing_host_pythonw(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"x")
    (scripts / "pythonw.exe").write_bytes(b"x")
    home = tmp_path / "conda"
    home.mkdir()
    (venv / "pyvenv.cfg").write_text(
        f"home = {home}\ninclude-system-site-packages = false\nversion = 3.13.12\n",
        encoding="ascii",
    )
    chosen = _ps(
        f"$p = Resolve-VtmDeskPython -VenvDir '{venv}' -PythonExe '{scripts / 'python.exe'}'; Write-Output $p"
    )
    assert chosen.lower().endswith("python.exe")
    assert chosen.lower().endswith("pythonw.exe") is False


def test_resolve_desk_python_uses_pythonw_when_host_has_it(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"x")
    (scripts / "pythonw.exe").write_bytes(b"x")
    home = tmp_path / "python"
    home.mkdir()
    (home / "pythonw.exe").write_bytes(b"x")
    (venv / "pyvenv.cfg").write_text(
        f"home = {home}\ninclude-system-site-packages = false\nversion = 3.13.7\n",
        encoding="ascii",
    )
    chosen = _ps(
        f"$p = Resolve-VtmDeskPython -VenvDir '{venv}' -PythonExe '{scripts / 'python.exe'}'; Write-Output $p"
    )
    assert chosen.lower().endswith("pythonw.exe")
