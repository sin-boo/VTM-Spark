"""Install / locate the bundled 'VTM Studio Cam' DirectShow virtual camera."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .paths import package_root

DEVICE_NAME = "VTM Studio Cam"

# Shown before the UAC prompt so it is clear what the admin rights are for.
ADMIN_NOTICE = (
    f"{DEVICE_NAME} needs a one-time setup.\n\n"
    "Windows will now ask for administrator permission. This is only for the "
    "virtual camera: it lets OBS, Discord, Zoom and other apps use VTM Studio "
    "as a webcam, and Windows only allows adding a camera as admin.\n\n"
    'The prompt may say "Windows Command Processor" - that is this step. '
    "Click Yes.\n\n"
    "You are only asked once."
)


def vcam_bundle_dir() -> Path:
    return package_root() / "vendor" / "tools" / "vtm_studio_cam"


def filter_dlls() -> list[Path]:
    d = vcam_bundle_dir()
    return [
        d / "UnityCaptureFilter64.dll",
        d / "UnityCaptureFilter32.dll",
    ]


def install_script() -> Path:
    return vcam_bundle_dir() / "Install-VTMStudioCam.bat"


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


def confirm_admin_prompt() -> bool:
    """Explain the coming UAC prompt in a dialog. False = the user cancelled."""
    if not sys.platform.startswith("win"):
        return True
    try:
        import ctypes

        MB_OKCANCEL = 0x1
        MB_ICONINFORMATION = 0x40
        MB_SETFOREGROUND = 0x10000
        MB_TOPMOST = 0x40000
        IDOK = 1
        flags = MB_OKCANCEL | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
        return ctypes.windll.user32.MessageBoxW(None, ADMIN_NOTICE, "VTM Studio", flags) == IDOK
    except Exception:
        return True


def ensure_installed(*, allow_prompt: bool = True) -> None:
    """Register VTM Studio Cam if missing. May show a UAC prompt once."""
    if device_available():
        return
    missing = [p for p in filter_dlls() if not p.is_file()]
    if missing:
        raise RuntimeError(
            "VTM Studio Cam filters missing — re-run install.bat "
            f"(expected under {vcam_bundle_dir()})"
        )
    script = install_script()
    if not script.is_file():
        raise RuntimeError(f"Installer missing: {script}")

    if not allow_prompt:
        raise RuntimeError(
            f"{DEVICE_NAME} is not installed. Click Virtual camera to add it "
            "(Windows asks for admin once)."
        )

    if not confirm_admin_prompt():
        raise RuntimeError(
            f"{DEVICE_NAME} setup cancelled. Click Virtual camera again when you are ready."
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
            f"Timed out installing {DEVICE_NAME}. Click Yes on the Windows prompt, "
            "then click Virtual camera again."
        ) from exc

    # UAC-elevated child may return before regsvr32 finishes; poll briefly.
    for _ in range(20):
        if device_available():
            return
        time.sleep(0.35)

    detail = (proc.stderr or proc.stdout or "").strip()
    hint = detail or f"exit {proc.returncode}"
    raise RuntimeError(
        f"Could not install {DEVICE_NAME}. Click Virtual camera again and choose "
        f"Yes on the Windows prompt. ({hint})"
    )

