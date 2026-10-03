"""The desk's per-picture CPU work outside the model: vcam, frame events, mids."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
from PIL import Image

from backend import frame_interp
from backend.stream import StreamRuntime
from backend.virtual_cam import VirtualCameraOut, trim_solid_edges


def _counting_as_rgb(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = VirtualCameraOut._as_rgb

    def spy(image, width, height):
        calls.append(1)
        return real(image, width, height)

    monkeypatch.setattr(VirtualCameraOut, "_as_rgb", staticmethod(spy))
    return calls


def test_vcam_reuses_the_prepared_frame_for_the_same_picture(monkeypatch) -> None:
    calls = _counting_as_rgb(monkeypatch)
    cam = VirtualCameraOut()
    still = Image.new("RGB", (32, 32), (12, 34, 56))
    first = cam._prepared(still, 32, 32)
    for _ in range(5):
        assert cam._prepared(still, 32, 32) is first
    assert len(calls) == 1


def test_vcam_redoes_the_frame_for_a_new_picture_or_size(monkeypatch) -> None:
    calls = _counting_as_rgb(monkeypatch)
    cam = VirtualCameraOut()
    a = Image.new("RGB", (32, 32), (10, 20, 30))
    b = Image.new("RGB", (32, 32), (200, 100, 50))
    np.testing.assert_array_equal(cam._prepared(a, 32, 32)[0, 0], (10, 20, 30))
    np.testing.assert_array_equal(cam._prepared(b, 32, 32)[0, 0], (200, 100, 50))
    assert cam._prepared(b, 16, 16).shape == (16, 16, 3)
    assert len(calls) == 3


def test_vcam_never_caches_arrays(monkeypatch) -> None:
    """An array can be drawn into in place; the same object is not the same picture."""
    calls = _counting_as_rgb(monkeypatch)
    cam = VirtualCameraOut()
    arr = np.full((32, 32, 3), 40, dtype=np.uint8)
    cam._prepared(arr, 32, 32)
    arr[:] = 220
    np.testing.assert_array_equal(cam._prepared(arr, 32, 32)[0, 0], (220, 220, 220))
    assert len(calls) == 2


def test_vcam_stop_drops_the_prepared_frame() -> None:
    cam = VirtualCameraOut()
    cam._prepared(Image.new("RGB", (32, 32)), 32, 32)
    cam.stop()
    assert cam._prep_src is None and cam._prep_out is None


def test_pump_keeps_sending_and_follows_a_new_picture(monkeypatch) -> None:
    calls = _counting_as_rgb(monkeypatch)
    sent: list[np.ndarray] = []

    class FakeCam:
        width = height = 32
        fps = 30
        device = "VTM Spark"
        backend = "unitycapture"

        def send(self, frame: np.ndarray) -> None:
            sent.append(np.asarray(frame).copy())

        def sleep_until_next_frame(self) -> None:
            time.sleep(0.005)

        def close(self) -> None:
            pass

    a = Image.new("RGB", (32, 32), (1, 2, 3))
    b = Image.new("RGB", (32, 32), (9, 8, 7))
    current = {"img": a}
    vcam = VirtualCameraOut()
    vcam._cam = FakeCam()
    vcam._width = vcam._height = 32
    vcam._fps = 30
    vcam._source = lambda: current["img"]
    vcam._start_pump()
    try:
        deadline = time.time() + 1.0
        while len(sent) < 4 and time.time() < deadline:
            time.sleep(0.01)
        current["img"] = b
        n = len(sent)
        while len(sent) < n + 4 and time.time() < deadline + 1.0:
            time.sleep(0.01)
    finally:
        vcam.stop()
    # Unity Capture needs a send every tick, but each picture is prepared once.
    assert len(sent) >= 8
    assert len(calls) == 2
    np.testing.assert_array_equal(sent[0][0, 0], (1, 2, 3))
    np.testing.assert_array_equal(sent[-1][0, 0], (9, 8, 7))


def test_trim_leaves_a_frame_lit_on_every_edge_untouched() -> None:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[0, 5] = arr[-1, 7] = arr[9, 0] = arr[11, -1] = (0, 0, 90)
    assert trim_solid_edges(arr) is arr


def test_trim_still_drops_a_one_sided_bar() -> None:
    arr = np.full((40, 40, 3), 120, dtype=np.uint8)
    arr[:, :8] = 0
    out = trim_solid_edges(arr)
    assert out.shape == (40, 32, 3)


def _payload_runtime(status: dict) -> StreamRuntime:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._listeners = []
    rt._status = status

    def no_status():
        raise AssertionError("a frame event must not build the full status")

    rt.status = no_status  # type: ignore[method-assign]
    rt._travel_preview_kps = None
    rt._body_lost = False
    rt._mouth_snapped = False
    return rt


def test_frame_payload_has_what_the_desk_reads_without_full_status() -> None:
    rt = _payload_runtime({"show_mesh": False, "show_hair": False})
    kps = np.zeros((37, 4), dtype=np.float32)
    payload = StreamRuntime._frame_payload(rt, Image.new("RGB", (16, 8), (5, 6, 7)), kps)
    assert payload["width"] == 16 and payload["height"] == 8
    assert isinstance(payload["image"], str) and payload["image"]
    assert len(payload["keypoints"]) == 37


def test_jpeg_of_a_non_rgb_picture_still_encodes() -> None:
    import base64

    from backend.stream import _image_to_jpeg_b64

    raw = base64.b64decode(_image_to_jpeg_b64(Image.new("RGBA", (8, 8), (1, 2, 3, 128))))
    assert raw[:2] == b"\xff\xd8"


def _display_runtime(wanted: bool) -> tuple[StreamRuntime, list[dict]]:
    rt = _payload_runtime({"show_mesh": False, "show_hair": False, "gen_fps": 10.0})
    events: list[dict] = []
    rt._listeners = [events.append]
    rt._last_overlay_kps = None
    rt._vcam_wanted = False
    rt._first_frame_pending = False
    rt._stream_status_t = time.perf_counter()  # no status push in this test
    rt.set_frame_demand(lambda: wanted)
    return rt, events


def test_shown_picture_skips_the_jpeg_when_nobody_can_see_it(monkeypatch) -> None:
    rt, events = _display_runtime(False)
    pushed: list[Image.Image] = []
    rt._vcam_wanted = True
    rt._push_virtual_cam = pushed.append  # type: ignore[method-assign]
    rt._frame_payload = lambda *_a: pytest.fail("encoded a frame nobody sees")  # type: ignore[method-assign]
    image = Image.new("RGB", (8, 8))
    StreamRuntime._publish_display_frame(rt, image, None, key=True)
    assert events == []
    # The virtual camera and "current picture" still follow the stream.
    assert pushed == [image]
    assert rt._last_image is image


def test_shown_picture_is_sent_when_the_desk_is_watching() -> None:
    rt, events = _display_runtime(True)
    StreamRuntime._publish_display_frame(rt, Image.new("RGB", (8, 8)), None, key=True)
    assert [e["type"] for e in events] == ["frame"]
    assert events[0]["image"]


def test_frames_wanted_defaults_on_without_a_demand_hook() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    assert StreamRuntime._frames_wanted(rt) is True
    rt._frame_demand = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert StreamRuntime._frames_wanted(rt) is True


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


def test_api_wants_frames_only_with_a_visible_client(monkeypatch) -> None:
    from backend import api

    monkeypatch.setattr(api, "_ws_clients", [])
    monkeypatch.setattr(api, "_ws_hidden", set())
    assert api.frames_wanted() is False
    ws = _FakeSocket()
    api._ws_clients.append(ws)
    assert api.frames_wanted() is True

    class Rt:
        def current_frame_event(self) -> dict:
            return {"type": "frame", "image": "now"}

    asyncio.run(api._note_visibility(ws, Rt(), hidden=True))
    assert api.frames_wanted() is False
    assert ws.sent == []
    # Back in view: the picture on the desk now, then live frames again.
    asyncio.run(api._note_visibility(ws, Rt(), hidden=False))
    assert api.frames_wanted() is True
    assert ws.sent == [{"type": "frame", "image": "now"}]
    # Already visible: no extra frame.
    asyncio.run(api._note_visibility(ws, Rt(), hidden=False))
    assert len(ws.sent) == 1
    with api._ws_lock:
        api._drop_client(ws)
    assert api._ws_hidden == set() and api.frames_wanted() is False


def test_lab_status_reuses_one_thread_pool() -> None:
    from backend import lab_harness
    from backend.lab_harness import LabHarness

    client = LabHarness()
    packets = {
        "/status": {"type": "status", "protocol": "track_lab.harness/1", "live": True},
        "/frame": {"type": "frame", "weights": {"A": 0.3}},
    }
    client._get = lambda path: packets[path]  # type: ignore[method-assign]
    first = client.status()
    pool = lab_harness._status_pool()
    second = client.status()
    assert lab_harness._status_pool() is pool
    assert first["weights"]["A"] == 0.3 and second["weights"]["A"] == 0.3


def test_warp_matches_the_per_axis_maps() -> None:
    import cv2

    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    flow = rng.normal(0, 3, (48, 64, 2)).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(48, dtype=np.float32))
    want = cv2.remap(
        image,
        (gx - flow[..., 0] * 0.4).astype(np.float32),
        (gy - flow[..., 1] * 0.4).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    got = frame_interp._warp_rgb(image, flow, 0.4)
    assert int(np.abs(got.astype(int) - want.astype(int)).max()) <= 1


def test_cv2_runs_on_a_few_threads() -> None:
    cv2 = frame_interp._cv2()
    assert cv2.getNumThreads() <= frame_interp.CV2_THREADS
