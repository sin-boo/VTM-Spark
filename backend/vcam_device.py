"""Install / locate the bundled 'VTM Noble Cam' DirectShow virtual camera."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .paths import package_root

DEVICE_NAME = "VTM Noble Cam"


def vcam_bundle_dir() -> Path:
    return package_root() / "vendor" / "tools" / "vtm_noble_cam"


def filter_dlls() -> list[Path]:
    d = vcam_bundle_dir()
    return [
        d / "UnityCaptureFilter64.dll",
        d / "UnityCaptureFilter32.dll",
    ]


def install_script() -> Path:
    return vcam_bundle_dir() / "Install-VTMNobleCam.bat"


def device_available() -> bool:
    """True if pyvirtualcam can open our named Unity Capture device."""
    try:
        import pyvirtualcam
    except ImportError:
        return False
    try:
        cam = pyvirtualcam.Camera(
            width=640,
            height=480,
            fps=10,
            backend="unitycapture",
            device=DEVICE_NAME,
            print_fps=False,
        )
        try:
            cam.close()
        except Exception:
            pass
        return True
    except Exception:
        return False


def ensure_installed(*, allow_prompt: bool = True) -> None:
    """Register VTM Noble Cam if missing. May show a UAC prompt once."""
    if device_available():
        return
    missing = [p for p in filter_dlls() if not p.is_file()]
    if missing:
        raise RuntimeError(
            "VTM Noble Cam filters missing — re-run Smart Build "
            f"(expected under {vcam_bundle_dir()})"
        )
    script = install_script()
    if not script.is_file():
        raise RuntimeError(f"Installer missing: {script}")

    if not allow_prompt:
        raise RuntimeError(
            f"{DEVICE_NAME} is not installed. Run Smart Build [1] "
            "(approve the UAC prompt once)."
        )

    # Elevated installer (UAC). Wait for registration to settle.
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            ["cmd.exe", "/c", str(script)],
            cwd=str(script.parent),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out installing {DEVICE_NAME}. Approve UAC if prompted."
        ) from exc

    # UAC-elevated child may return before regsvr32 finishes; poll briefly.
    for _ in range(20):
        if device_available():
            return
        time.sleep(0.35)

    detail = (proc.stderr or proc.stdout or "").strip()
    hint = detail or f"exit {proc.returncode}"
    raise RuntimeError(
        f"Could not install {DEVICE_NAME}. Approve UAC when prompted, "
        f"then retry Virtual camera. ({hint})"
    )

