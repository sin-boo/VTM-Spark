"""Webcam capture must not take the Windows microphone."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
from cameras import CameraInfo
from cameras import _is_camera_name as poser_is_camera_name
from cameras import _no_mic_params as poser_no_mic_params
from cameras import _open_cv_index
from live_poser import pick_default_camera

# Track Lab imports its ``harness`` package top-level (its pytest.ini runs
# from track_lab/). Append so the desk's own ``backend`` still wins.
_LAB = Path(__file__).resolve().parents[2] / "track_lab"
if str(_LAB) not in sys.path:
    sys.path.append(str(_LAB))

from track_lab.backend import cameras as lab_cameras  # noqa: E402
from track_lab.backend.osf_cam import OsfCam  # noqa: E402


def _assert_no_mic_params(params: list[int]) -> None:
    audio = int(cv2.CAP_PROP_AUDIO_STREAM)
    assert audio in params
    assert params[params.index(audio) + 1] == -1
    video = getattr(cv2, "CAP_PROP_VIDEO_STREAM", None)
    if video is not None:
        assert int(video) in params
        assert params[params.index(int(video)) + 1] == 0


def test_no_mic_params_disable_audio_stream() -> None:
    _assert_no_mic_params(lab_cameras._no_mic_params())
    _assert_no_mic_params(poser_no_mic_params())


def test_open_video_passes_audio_stream_off(monkeypatch) -> None:
    opened: list[tuple] = []

    class FakeCap:
        def open(self, *args) -> bool:
            opened.append(args)
            return False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(lab_cameras.cv2, "VideoCapture", lambda: FakeCap())
    cap = lab_cameras.open_video(0, cv2.CAP_DSHOW)
    assert cap is not None
    assert opened
    args = opened[0]
    assert args[0] == 0
    params = list(args[2])
    _assert_no_mic_params(params)


def test_poser_open_passes_audio_and_video_stream(monkeypatch) -> None:
    opened: list[tuple] = []

    class FakeCap:
        def open(self, *args) -> bool:
            opened.append(args)
            return False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    import cameras as poser_cameras

    monkeypatch.setattr(poser_cameras.cv2, "VideoCapture", lambda: FakeCap())
    cap = _open_cv_index(0, cv2.CAP_DSHOW)
    assert cap is not None
    assert opened
    _assert_no_mic_params(list(opened[0][2]))


def test_capture_sources_disable_audio_stream() -> None:
    root = Path(__file__).resolve().parents[2]
    lab = (root / "track_lab" / "backend" / "cameras.py").read_text(encoding="utf-8")
    poser = (root / "vendor" / "tools" / "live-poser" / "cameras.py").read_text(
        encoding="utf-8"
    )
    assert "CAP_PROP_AUDIO_STREAM" in lab
    assert "CAP_PROP_AUDIO_STREAM" in poser
    assert "CAP_PROP_VIDEO_STREAM" in lab
    assert "CAP_PROP_VIDEO_STREAM" in poser
    assert "cv2.VideoCapture(i" not in lab
    assert "cv2.VideoCapture(i" not in poser
    assert "cv2.VideoCapture(int(index)" not in lab
    assert "cv2.VideoCapture(camera_index" not in poser


def test_dshow_list_skips_microphone_devices() -> None:
    for is_cam in (lab_cameras._is_camera_name, poser_is_camera_name):
        assert is_cam("Integrated Camera") is True
        assert is_cam("Microphone (USB Audio)") is False
        assert is_cam("Stereo Mix") is False
        assert is_cam("OBS Virtual Camera") is True


def test_preferred_camera_skips_usb_microphone() -> None:
    cams = [
        CameraInfo(index=0, name="Microphone (USB Audio)", backend="dshow"),
        CameraInfo(index=1, name="Integrated Camera", backend="dshow"),
    ]
    assert pick_default_camera(cams) == 1


def test_osf_stop_releases_hung_capture() -> None:
    class FakeCap:
        def __init__(self) -> None:
            self.released = False
            self._block = threading.Event()

        def read(self):
            self._block.wait(timeout=30)
            return False, None

        def release(self) -> None:
            self.released = True
            self._block.set()

    cam = OsfCam()
    cap = FakeCap()
    cam._cap = cap
    cam._running = True

    def loop() -> None:
        held = cam._cap
        try:
            while cam._running:
                if held is None:
                    break
                held.read()
        finally:
            cam._release()

    thread = threading.Thread(target=loop, daemon=True, name="osf-cam-test")
    cam._thread = thread
    thread.start()
    time.sleep(0.05)
    cam.stop()
    assert cap.released is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()
