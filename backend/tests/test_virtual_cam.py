from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.stream import StreamRuntime
from backend.virtual_cam import (
    VirtualCameraOut,
    cover_rgb,
    trim_solid_edges,
    vcam_even_size,
)


def test_vcam_even_size_steps_of_four() -> None:
    assert vcam_even_size(768, 768) == (768, 768)
    assert vcam_even_size(1920, 1080) == (1920, 1080)
    assert vcam_even_size(770, 771) == (768, 770)


def test_cover_fills_square_from_widescreen() -> None:
    src = np.full((108, 192, 3), (200, 10, 10), dtype=np.uint8)
    out = cover_rgb(src, 64, 64)
    assert out.shape == (64, 64, 3)
    np.testing.assert_array_equal(out[0, 0], (200, 10, 10))
    np.testing.assert_array_equal(out[32, 32], (200, 10, 10))


def test_trim_drops_black_letterbox() -> None:
    arr = np.zeros((100, 80, 3), dtype=np.uint8)
    arr[20:80, :, :] = (40, 200, 50)
    cropped = trim_solid_edges(arr)
    assert cropped.shape == (60, 80, 3)
    np.testing.assert_array_equal(cropped[0, 0], (40, 200, 50))


def test_as_rgb_trims_then_covers_to_square() -> None:
    arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    arr[180:900, :, :] = (30, 190, 40)
    out = VirtualCameraOut._as_rgb(arr, 768, 768)
    assert out.shape == (768, 768, 3)
    assert int(out.mean()) > 40
    # Interior is the green field, not the black bars.
    np.testing.assert_array_equal(out[384, 384], (30, 190, 40))


def test_as_rgb_keeps_matching_square() -> None:
    img = Image.new("RGB", (64, 64), (9, 8, 7))
    out = VirtualCameraOut._as_rgb(img, 64, 64)
    assert out.shape == (64, 64, 3)
    np.testing.assert_array_equal(out[0, 0], (9, 8, 7))


def test_vcam_frame_size_ignores_widescreen_still() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(image_size=768)
    rt._last_image = Image.new("RGB", (1920, 1080), (0, 0, 0))
    assert StreamRuntime._vcam_frame_size(rt) == (768, 768)


def test_vcam_frame_size_uses_square_generated() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(image_size=512)
    rt._last_image = Image.new("RGB", (768, 768), (1, 2, 3))
    assert StreamRuntime._vcam_frame_size(rt) == (768, 768)


def test_pump_resends_reference_still() -> None:
    vcam = VirtualCameraOut()
    sent: list[np.ndarray] = []

    class FakeCam:
        width = 32
        height = 32
        fps = 30
        device = "VTM Noble Cam"
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
