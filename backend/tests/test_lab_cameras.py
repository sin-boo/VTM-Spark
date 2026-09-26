"""Camera open must survive virtual cams, not only a built-in webcam."""

from __future__ import annotations

import numpy as np

from track_lab.backend import cameras as lab


def test_pick_default_uses_last_camera_name_when_index_moves() -> None:
    cams = [
        {"index": 0, "name": "nizima LIVE Virtual Camera"},
        {"index": 2, "name": "DroidCam Video"},
    ]
    assert lab.pick_default(cams, 1, saved_name="DroidCam Video") == 2


def test_pick_default_keeps_saved_index_without_a_name() -> None:
    cams = [
        {"index": 0, "name": "nizima LIVE Virtual Camera"},
        {"index": 1, "name": "DroidCam Video"},
    ]
    assert lab.pick_default(cams, 1) == 1


def test_save_remembers_camera_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lab, "CAM_SAVE", tmp_path / "camera.json")
    lab.save_camera_index(1, "DroidCam Video")
    assert lab.load_camera_choice() == (1, "DroidCam Video")
    lab.save_camera_index(1)
    assert lab.load_camera_choice() == (1, "DroidCam Video")


def test_rank_dcaps_prefers_droidcam_mid_res() -> None:
    caps = [
        {"id": 1, "minCX": 1280, "minCY": 720, "minInterval": 333333},
        {"id": 2, "minCX": 640, "minCY": 480, "minInterval": 333333},
        {"id": 3, "minCX": 640, "minCY": 480, "minInterval": 666666},
    ]
    assert lab._rank_dcaps(caps, 640, 480, prefer_mid=True)[0] == 2


def test_open_waits_for_first_virtual_frame(monkeypatch) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0, 0] = 255

    class Slow:
        def __init__(self) -> None:
            self.reads = 0

        def open(self, *args) -> bool:
            return True

        def isOpened(self) -> bool:
            return True

        def set(self, *args) -> bool:
            return True

        def read(self):
            self.reads += 1
            if self.reads < 3:
                return False, None
            return True, frame

        def release(self) -> None:
            return None

    monkeypatch.setattr(lab, "_open_dshow_reader", lambda *args, **kwargs: None)
    monkeypatch.setattr(lab.cv2, "VideoCapture", lambda: Slow())
    monkeypatch.setattr(lab, "_opencv_backends", lambda: [1])
    monkeypatch.setattr(lab.time, "sleep", lambda *_a, **_k: None)

    cap = lab.open_capture(1, 640, 480)
    assert cap is not None
    assert cap.read()[0] is True


def test_open_tries_next_backend_when_dshow_rejects(monkeypatch) -> None:
    frame = np.full((4, 4, 3), 40, dtype=np.uint8)
    opened: list[int] = []

    class Gate:
        def __init__(self) -> None:
            self.backend = None

        def open(self, index, backend, *args) -> bool:
            self.backend = int(backend)
            opened.append(int(backend))
            return True

        def isOpened(self) -> bool:
            return self.backend == 2

        def set(self, *args) -> bool:
            return True

        def read(self):
            return True, frame

        def release(self) -> None:
            return None

    monkeypatch.setattr(lab, "_open_dshow_reader", lambda *args, **kwargs: None)
    monkeypatch.setattr(lab.cv2, "VideoCapture", lambda: Gate())
    monkeypatch.setattr(lab, "_opencv_backends", lambda: [1, 2])
    monkeypatch.setattr(lab.time, "sleep", lambda *_a, **_k: None)

    cap = lab.open_capture(1, 640, 480)
    assert cap is not None
    assert opened[0] == 1
    assert 2 in opened


def test_open_uses_native_mode_when_forced_size_is_empty(monkeypatch) -> None:
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)

    class Sized:
        def __init__(self) -> None:
            self.w = 0
            self.h = 0

        def open(self, *args) -> bool:
            return True

        def isOpened(self) -> bool:
            return True

        def set(self, prop, value) -> bool:
            if prop == lab.cv2.CAP_PROP_FRAME_WIDTH:
                self.w = int(value)
            elif prop == lab.cv2.CAP_PROP_FRAME_HEIGHT:
                self.h = int(value)
            return True

        def read(self):
            if self.w == 0 and self.h == 0:
                return True, frame
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(lab, "_open_dshow_reader", lambda *args, **kwargs: None)
    monkeypatch.setattr(lab.cv2, "VideoCapture", lambda: Sized())
    monkeypatch.setattr(lab, "_opencv_backends", lambda: [1])
    monkeypatch.setattr(lab.time, "sleep", lambda *_a, **_k: None)

    cap = lab.open_capture(1, 640, 480)
    assert cap is not None
    ok, got = cap.read()
    assert ok and got is frame


def test_mic_params_fall_back_when_virtual_cam_rejects_them(monkeypatch) -> None:
    calls: list[tuple] = []

    class Picky:
        def __init__(self) -> None:
            self._open = False

        def open(self, *args) -> bool:
            calls.append(args)
            self._open = len(args) == 2
            return self._open

        def isOpened(self) -> bool:
            return self._open

        def release(self) -> None:
            self._open = False

    monkeypatch.setattr(lab.cv2, "VideoCapture", lambda: Picky())
    cap = lab.open_video(1, 700)
    assert cap.isOpened()
    assert len(calls[0]) == 3
    assert len(calls[1]) == 2
