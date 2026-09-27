import threading

from PIL import Image

from backend.stream import StreamRuntime


def _bare_stream() -> StreamRuntime:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._listeners = []
    rt._streaming = True
    rt._paused = False
    rt._frame_in_flight = False
    rt._status = {
        "streaming": True,
        "paused": False,
        "message": "Streaming",
        "gen_fps": 0.0,
        "busy": False,
        "interpolate": True,
        "inbetweens": 1,
    }
    rt._inbetween_prev = None
    rt._inbetween_prev_kps = None
    rt._display_busy = False
    rt._last_interp_s = 0.0
    rt._display_queue = __import__("queue").Queue()
    rt._last_display_t = 0.0
    rt._last_key_t = 0.0
    rt._first_frame_pending = False
    return rt


def test_pause_stream_holds_and_resume_reschedules() -> None:
    rt = _bare_stream()
    scheduled: list[int] = []
    rt._schedule_next_frame = lambda: scheduled.append(1)  # type: ignore[method-assign]
    StreamRuntime.pause_stream(rt)
    assert rt._paused is True
    assert rt._status["paused"] is True
    assert rt._status["message"] == "Stream paused"
    StreamRuntime.pause_stream(rt)
    assert scheduled == []
    StreamRuntime.resume_stream(rt)
    assert rt._paused is False
    assert rt._status["paused"] is False
    assert scheduled == [1]


def test_on_frame_starts_next_gen_before_display() -> None:
    rt = _bare_stream()
    rt._frame_in_flight = True
    order: list[str] = []
    rt._enqueue_display = lambda item: order.append("display")  # type: ignore[method-assign]
    rt._schedule_next_frame = lambda: order.append("next")  # type: ignore[method-assign]
    rt._emit = lambda ev: None  # type: ignore[method-assign]
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
    image = Image.new("RGB", (4, 4), (8, 8, 8))
    StreamRuntime._on_frame(
        rt, image, 0.05, {}, None, streaming=True, schedule_next=True
    )
    assert order == ["next", "display"]
    assert rt._frame_in_flight is False


def test_enqueue_display_skips_mids_when_busy() -> None:
    rt = _bare_stream()
    rt._display_busy = True
    rt._status["gen_fps"] = 10.0
    StreamRuntime._enqueue_display(
        rt,
        {
            "image": Image.new("RGB", (4, 4), (1, 1, 1)),
            "keypoints": None,
            "prev": Image.new("RGB", (4, 4), (0, 0, 0)),
            "prev_kps": None,
            "count": 1,
        },
    )
    job = rt._display_queue.get_nowait()
    assert job["count"] == 0
    assert job["prev"] is None


def test_enqueue_display_keeps_every_key() -> None:
    rt = _bare_stream()
    rt._display_busy = True
    first = Image.new("RGB", (4, 4), (1, 1, 1))
    second = Image.new("RGB", (4, 4), (2, 2, 2))
    StreamRuntime._enqueue_display(
        rt,
        {
            "image": first,
            "keypoints": None,
            "prev": Image.new("RGB", (4, 4), (0, 0, 0)),
            "prev_kps": None,
            "count": 1,
        },
    )
    StreamRuntime._enqueue_display(
        rt,
        {
            "image": second,
            "keypoints": None,
            "prev": first,
            "prev_kps": None,
            "count": 1,
        },
    )
    assert rt._display_queue.qsize() == 2
    a = rt._display_queue.get_nowait()
    b = rt._display_queue.get_nowait()
    assert a["image"] is first
    assert b["image"] is second
    assert a["count"] == 0
    assert b["count"] == 0


def test_ema_fps_counts_keys_and_shown() -> None:
    rt = _bare_stream()
    rt._status["show_fps"] = 0.0
    rt._status["gen_fps"] = 0.0
    StreamRuntime._ema_fps(rt, "_last_display_t", "show_fps", 1.0)
    StreamRuntime._ema_fps(rt, "_last_key_t", "gen_fps", 1.0)
    shown = StreamRuntime._ema_fps(rt, "_last_display_t", "show_fps", 1.05)
    StreamRuntime._ema_fps(rt, "_last_display_t", "show_fps", 1.10)
    keys = StreamRuntime._ema_fps(rt, "_last_key_t", "gen_fps", 1.10)
    assert shown > keys
    assert keys > 0
