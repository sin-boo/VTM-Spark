from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.stream import StreamRuntime
from backend.virtual_cam import (
    VirtualCameraOut,
    fit_rgb,
    trim_solid_edges,
    vcam_even_size,
    vcam_wide_size,
)


def test_vcam_even_size_steps_of_four() -> None:
    assert vcam_even_size(768, 768) == (768, 768)
    assert vcam_even_size(1920, 1080) == (1920, 1080)
    assert vcam_even_size(770, 771) == (768, 770)


def test_wide_size_is_exact_16_by_9_around_the_square() -> None:
    assert vcam_wide_size(768) == (1376, 774)
    assert vcam_wide_size(512) == (928, 522)
    for side in (64, 500, 768, 1024):
        w, h = vcam_wide_size(side)
        assert h >= side and w * 9 == h * 16
        assert vcam_even_size(w, h) == (w, h)


def test_fit_pads_square_with_its_background_not_black() -> None:
    src = np.full((768, 768, 3), (40, 200, 50), dtype=np.uint8)
    src[300:500, 300:500] = (250, 220, 200)  # the character
    out = fit_rgb(src, 1376, 774)
    assert out.shape == (774, 1376, 3)
    np.testing.assert_array_equal(out[0, 0], (40, 200, 50))
    np.testing.assert_array_equal(out[400, 10], (40, 200, 50))
    np.testing.assert_array_equal(out[773, 1375], (40, 200, 50))
    # Unscaled and centered: the square starts at x=304, y=3.
    np.testing.assert_array_equal(out[3:771, 304:1072], src)


def test_fit_scales_widescreen_into_frame() -> None:
    src = np.full((1080, 1920, 3), (200, 10, 10), dtype=np.uint8)
    out = fit_rgb(src, 1376, 774)
    assert out.shape == (774, 1376, 3)
    np.testing.assert_array_equal(out[0, 0], (200, 10, 10))
    np.testing.assert_array_equal(out[387, 688], (200, 10, 10))


def test_trim_drops_black_letterbox() -> None:
    arr = np.zeros((100, 80, 3), dtype=np.uint8)
    arr[20:80, :, :] = (40, 200, 50)
    cropped = trim_solid_edges(arr)
    assert cropped.shape == (60, 80, 3)
    np.testing.assert_array_equal(cropped[0, 0], (40, 200, 50))


def test_as_rgb_trims_letterbox_then_fills_wide_frame() -> None:
    arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    arr[180:900, :, :] = (30, 190, 40)
    out = VirtualCameraOut._as_rgb(arr, 1376, 774)
    assert out.shape == (774, 1376, 3)
    # Green edge to edge, none of the black bars.
    assert int(out.max(axis=2).min()) > 16
    np.testing.assert_array_equal(out[387, 688], (30, 190, 40))


def test_as_rgb_keeps_matching_square() -> None:
    img = Image.new("RGB", (64, 64), (9, 8, 7))
    out = VirtualCameraOut._as_rgb(img, 64, 64)
    assert out.shape == (64, 64, 3)
    np.testing.assert_array_equal(out[0, 0], (9, 8, 7))


def test_vcam_frame_size_ignores_widescreen_still() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(image_size=768)
    rt._last_image = Image.new("RGB", (1920, 1080), (0, 0, 0))
    assert StreamRuntime._vcam_frame_size(rt) == (1376, 774)


def test_vcam_frame_size_uses_square_generated() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(image_size=512)
    rt._last_image = Image.new("RGB", (768, 768), (1, 2, 3))
    assert StreamRuntime._vcam_frame_size(rt) == (1376, 774)


def test_pump_resends_reference_still() -> None:
    vcam = VirtualCameraOut()
    sent: list[np.ndarray] = []

    class FakeCam:
        width = 32
        height = 32
        fps = 30
        device = "VTM Spark"
        backend = "unitycapture"

        def send(self, frame: np.ndarray) -> None:
            sent.append(np.asarray(frame).copy())

        def sleep_until_next_frame(self) -> None:
            time.sleep(0.01)

        def close(self) -> None:
            pass

    still = Image.new("RGB", (32, 32), (12, 34, 56))
    vcam._cam = FakeCam()
    vcam._width = 32
    vcam._height = 32
    vcam._fps = 30
    vcam._source = lambda: still
    vcam._start_pump()
    deadline = time.time() + 1.0
    while len(sent) < 3 and time.time() < deadline:
        time.sleep(0.02)
    vcam.stop()
    assert len(sent) >= 3
    assert sent[0].shape == (32, 32, 3)
    np.testing.assert_array_equal(sent[0][0, 0], (12, 34, 56))


def _vcam_ready_to_install(monkeypatch, tmp_path):
    from backend import vcam_device

    for name in ("UnityCaptureFilter64.dll", "UnityCaptureFilter32.dll", "VTM Spark Camera Setup.exe"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(vcam_device, "vcam_bundle_dir", lambda: tmp_path)
    monkeypatch.setattr(vcam_device, "device_available", lambda: False)
    ran: list[object] = []
    monkeypatch.setattr(vcam_device.subprocess, "run", lambda *a, **k: ran.append(a))
    return vcam_device, ran


def test_cancelled_camera_notice_skips_the_admin_prompt(monkeypatch, tmp_path) -> None:
    import pytest

    vcam_device, ran = _vcam_ready_to_install(monkeypatch, tmp_path)
    monkeypatch.setattr(vcam_device, "confirm_admin_prompt", lambda: False)
    with pytest.raises(RuntimeError, match="cancelled"):
        vcam_device.ensure_installed(allow_prompt=True)
    assert ran == []


def test_camera_notice_comes_before_the_admin_prompt(monkeypatch, tmp_path) -> None:
    vcam_device, ran = _vcam_ready_to_install(monkeypatch, tmp_path)
    order: list[str] = []
    monkeypatch.setattr(vcam_device, "confirm_admin_prompt", lambda: order.append("notice") or True)
    monkeypatch.setattr(
        vcam_device.subprocess, "run",
        lambda *a, **k: order.append("uac") or type("P", (), {"stderr": "", "stdout": "", "returncode": 1})(),
    )
    monkeypatch.setattr(vcam_device.time, "sleep", lambda _s: None)
    try:
        vcam_device.ensure_installed(allow_prompt=True)
    except RuntimeError:
        pass
    assert order == ["notice", "uac"]
    notice = vcam_device.ADMIN_NOTICE
    assert "administrator" in notice and "virtual camera" in notice and "webcam" in notice


def test_camera_is_named_vtm_spark() -> None:
    from backend import vcam_device

    assert vcam_device.DEVICE_NAME == "VTM Spark"
    assert vcam_device.setup_exe().is_file()
    src = (Path(__file__).resolve().parents[1] / "packaging" / "cam-setup.cs").read_text(encoding="utf-8")
    assert f'DeviceName = "{vcam_device.DEVICE_NAME}"' in src
    assert "UnityCaptureName=" in src


def _version_string(exe: Path, field: str) -> str:
    import ctypes

    ver = ctypes.windll.version
    size = ver.GetFileVersionInfoSizeW(str(exe), None)
    buf = ctypes.create_string_buffer(size)
    assert ver.GetFileVersionInfoW(str(exe), 0, size, buf)
    ptr = ctypes.c_void_p()
    n = ctypes.c_uint()
    assert ver.VerQueryValueW(buf, "\\VarFileInfo\\Translation", ctypes.byref(ptr), ctypes.byref(n))
    lang, page = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ushort * 2)).contents
    key = f"\\StringFileInfo\\{lang:04x}{page:04x}\\{field}"
    assert ver.VerQueryValueW(buf, key, ctypes.byref(ptr), ctypes.byref(n))
    return ctypes.wstring_at(ptr.value, n.value).rstrip("\0")


def test_admin_prompt_names_vtm_spark_not_the_command_processor() -> None:
    """The prompt shows the elevated program's name and icon. The .bat went
    through cmd.exe: "Windows Command Processor" and the console icon."""
    import sys

    import pytest

    from backend import vcam_device

    if not sys.platform.startswith("win"):
        pytest.skip("Windows version resources")
    exe = vcam_device.setup_exe()
    assert exe.stem == "VTM Spark Camera Setup"
    assert _version_string(exe, "FileDescription") == "VTM Spark Camera Setup"
    assert _version_string(exe, "ProductName") == "VTM Spark"
    menu = (Path(__file__).resolve().parents[1] / "packaging" / "start-menu.ps1").read_text(encoding="utf-8-sig")
    assert "VTM Spark Camera Setup.exe" in menu
    assert "Windows Command Processor" not in menu
    assert "Windows Command Processor" not in vcam_device.ADMIN_NOTICE


def test_declined_admin_prompt_says_so(monkeypatch, tmp_path) -> None:
    import pytest

    vcam_device, ran = _vcam_ready_to_install(monkeypatch, tmp_path)
    monkeypatch.setattr(vcam_device, "confirm_admin_prompt", lambda: True)
    monkeypatch.setattr(
        vcam_device.subprocess, "run",
        lambda *a, **k: ran.append(a[0])
        or type("P", (), {"stderr": "", "stdout": "", "returncode": vcam_device.SETUP_DECLINED})(),
    )
    with pytest.raises(RuntimeError, match="Yes on the Windows prompt"):
        vcam_device.ensure_installed(allow_prompt=True)
    # The setup exe itself, not cmd.exe running a .bat.
    assert ran == [[str(tmp_path / "VTM Spark Camera Setup.exe")]]


def test_stale_registration_is_not_ready(monkeypatch, tmp_path) -> None:
    from backend import vcam_device

    live = tmp_path / "UnityCaptureFilter64.dll"
    live.write_bytes(b"")
    gone = tmp_path / "moved" / "UnityCaptureFilter32.dll"
    monkeypatch.setattr(vcam_device.sys, "platform", "win32")
    monkeypatch.setattr(vcam_device, "installed_dir", lambda: tmp_path)
    monkeypatch.setattr(vcam_device, "device_available", lambda: True)
    monkeypatch.setattr(vcam_device, "registered_filters", lambda: [live, gone])
    assert vcam_device.registration_ok() is False
    assert vcam_device.device_ready() is False
    monkeypatch.setattr(vcam_device, "registered_filters", lambda: [live, None])
    assert vcam_device.registration_ok() is False
    monkeypatch.setattr(vcam_device, "registered_filters", lambda: [])
    assert vcam_device.registration_ok() is False
    monkeypatch.setattr(vcam_device, "registered_filters", lambda: [live])
    assert vcam_device.device_ready() is True


def test_camera_registered_from_an_app_folder_is_not_ready(monkeypatch, tmp_path) -> None:
    """Registered from the vendor folder, every program that lists webcams held
    the DLL open and the app folder could not be deleted."""
    from backend import vcam_device

    installed = tmp_path / "Program Files" / "VTM Spark" / "Camera"
    vendor = tmp_path / "VTM" / "vendor" / "tools" / "vtm_spark_cam"
    for d in (installed, vendor):
        d.mkdir(parents=True)
        (d / "UnityCaptureFilter64.dll").write_bytes(b"")
    monkeypatch.setattr(vcam_device.sys, "platform", "win32")
    monkeypatch.setattr(vcam_device, "installed_dir", lambda: installed)
    monkeypatch.setattr(vcam_device, "registered_filters", lambda: [vendor / "UnityCaptureFilter64.dll"])
    assert vcam_device.registration_ok() is False
    upper = Path(str(installed).upper()) / "UnityCaptureFilter64.dll"
    monkeypatch.setattr(vcam_device, "registered_filters", lambda: [upper])
    assert vcam_device.registration_ok() is True


def test_camera_setup_registers_a_copy_outside_the_app() -> None:
    src = (Path(__file__).resolve().parents[1] / "packaging" / "cam-setup.cs").read_text(encoding="utf-8")
    assert "ProgramW6432" in src and '"VTM Spark"), "Camera")' in src
    assert "dll64 = Place(dll64, dir);" in src
    # A copy some program already holds moves out of the app folder, so the
    # folder deletes without a restart.
    assert "Release(old, dir);" in src and "MoveFileDelayUntilReboot" in src


def test_uninstall_bat_removes_the_camera_through_the_setup_exe() -> None:
    root = Path(__file__).resolve().parents[2]
    bat = (root / "uninstall.bat").read_text(encoding="utf-8")
    assert "backend\\packaging\\uninstall.ps1" in bat
    # Exit 3 = delete the folder, from the last line once cmd has left it.
    last = bat.strip().splitlines()[-1]
    assert '"%EC%"=="3"' in bat and "(goto) 2>nul & rd /s /q \"%ROOT%\"" in last
    script = (root / "backend" / "packaging" / "uninstall.ps1").read_text(encoding="utf-8")
    assert "exit 3" in script
    assert "VTM Spark Camera Setup.exe" in script and '"--uninstall"' in script
    assert '".venv-build"' in script and '".tools"' in script
    src = (root / "backend" / "packaging" / "cam-setup.cs").read_text(encoding="utf-8")
    # The elevated relaunch keeps --uninstall, and it unregisters with regsvr32 /u.
    assert "ElevatedFlag + \" \" + UninstallFlag" in src
    assert '"/u /s \\""' in src


def test_update_bat_pulls_with_git_or_downloads_the_zip() -> None:
    root = Path(__file__).resolve().parents[2]
    bat = (root / "update.bat").read_text(encoding="utf-8")
    assert "backend\\packaging\\update.ps1" in bat
    # One parenthesised block, so replacing update.bat mid-run is safe.
    assert bat.index("(") < bat.index("update.ps1") < bat.rindex(")")
    script = (root / "backend" / "packaging" / "update.ps1").read_text(encoding="utf-8")
    assert "pull --ff-only" in script
    assert "github.com/$Repo/archive/$ref.zip" in script
    assert "stop-app.ps1" in script and "install.bat" in script
    # git's own output must not become Update-WithGit's return value.
    assert "| Out-Host" in script


def test_stale_registration_reinstalls_even_when_the_device_opens(monkeypatch, tmp_path) -> None:
    vcam_device, ran = _vcam_ready_to_install(monkeypatch, tmp_path)
    monkeypatch.setattr(vcam_device, "device_available", lambda: True)
    monkeypatch.setattr(vcam_device, "registration_ok", lambda: False)
    monkeypatch.setattr(vcam_device, "confirm_admin_prompt", lambda: True)
    monkeypatch.setattr(
        vcam_device.subprocess, "run",
        lambda *a, **k: ran.append(a) or type("P", (), {"stderr": "", "stdout": "", "returncode": 1})(),
    )
    monkeypatch.setattr(vcam_device.time, "sleep", lambda _s: None)
    try:
        vcam_device.ensure_installed(allow_prompt=True)
    except RuntimeError:
        pass
    assert len(ran) == 1
