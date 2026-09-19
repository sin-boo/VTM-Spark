import subprocess
from pathlib import Path


def _pack() -> Path:
    return Path(__file__).resolve().parents[1] / "packaging"


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
