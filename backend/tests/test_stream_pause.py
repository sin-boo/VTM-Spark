import threading

import numpy as np

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


def _img(v: int) -> Image.Image:
    return Image.new("RGB", (4, 4), (v, v, v))


def test_enqueue_display_keeps_every_key() -> None:
    """A job arriving while the last is still on screen is normal, not a
    backlog: it keeps its mids (they used to be stripped here)."""
    rt = _bare_stream()
    rt._display_busy = True
    first = {"keys": [(_img(1), None)], "prev": _img(0), "prev_kps": None, "count": 1}
    second = {"keys": [(_img(2), None)], "prev": _img(1), "prev_kps": None, "count": 1}
    StreamRuntime._enqueue_display(rt, first)
    StreamRuntime._enqueue_display(rt, second)
    assert rt._display_queue.qsize() == 2
    assert rt._display_queue.get_nowait()["keys"] is first["keys"]
    assert rt._display_queue.get_nowait()["count"] == 1


def test_display_is_behind_only_when_really_late() -> None:
    """A big batch plays over several key intervals; the next call landing a
    few ms early is on schedule, not a backlog (it used to drop mids)."""
    import time

    rt = _bare_stream()
    job = {"keys": [(_img(1), None)], "prev": None, "prev_kps": None, "count": 1}
    assert StreamRuntime._display_behind(rt, 0.1) is False
    StreamRuntime._enqueue_display(rt, job)
    assert StreamRuntime._display_behind(rt, 0.1) is False
    # Waited longer than a key interval: late.
    with rt._display_queue.mutex:
        rt._display_queue.queue[0]["queued_at"] = time.perf_counter() - 0.2
    assert StreamRuntime._display_behind(rt, 0.1) is True
    # Two calls waiting: late whatever the clock says.
    rt2 = _bare_stream()
    StreamRuntime._enqueue_display(rt2, dict(job))
    StreamRuntime._enqueue_display(rt2, dict(job))
    assert StreamRuntime._display_behind(rt2, 0.1) is True


def test_batch_keys_queue_as_one_job() -> None:
    """Batch×2 as two jobs made the second look queued, so no mid was ever
    drawn and the pair flashed 50 ms apart."""
    rt = _bare_stream()
    rt._emit = lambda ev: None  # type: ignore[method-assign]
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
    rt._schedule_next_frame = lambda: None  # type: ignore[method-assign]
    before = _img(0)
    rt._inbetween_prev = before
    a, b = _img(5), _img(9)
    StreamRuntime._on_stream_keys(rt, [a, b], [None, None], 0.06, {})
    assert rt._display_queue.qsize() == 1
    job = rt._display_queue.get_nowait()
    assert [k[0] for k in job["keys"]] == [a, b]
    assert job["prev"] is before
    assert job["count"] == 1


def test_play_job_puts_mids_between_every_key() -> None:
    rt = _bare_stream()
    shown: list[tuple[int, bool]] = []
    paced: list[float] = []
    rt._pace_display = lambda gap=0.0: paced.append(gap)  # type: ignore[method-assign]
    rt._publish_display_frame = (  # type: ignore[method-assign]
        lambda image, kps, key=True: shown.append((int(np.asarray(image).mean()), key))
    )
    job = {
        "keys": [(_img(100), None), (_img(200), None)],
        "prev": _img(0),
        "prev_kps": None,
        "count": 1,
        # Batch×2 at 5 calls/s = 10 keys/s.
        "key_interval": 0.1,
    }
    StreamRuntime._play_display_job(rt, job)
    assert [k for _, k in shown] == [False, True, False, True]
    levels = [v for v, _ in shown]
    assert levels == sorted(levels)
    # Four pictures evenly over the call's 200 ms.
    assert all(abs(g - 0.0475) < 1e-6 for g in paced)


def test_gen_fps_comes_from_call_timing() -> None:
    rt = _bare_stream()
    StreamRuntime._reset_display_clock(rt)
    StreamRuntime._note_key_interval(rt, 10.0, 2)
    StreamRuntime._note_key_interval(rt, 10.2, 2)
    # Two keys every 200 ms = 10 keys/s, however the pair is shown.
    assert abs(rt._status["gen_fps"] - 10.0) < 1e-6
    # A stall (or pause) is not the rate.
    StreamRuntime._note_key_interval(rt, 20.0, 2)
    assert abs(rt._status["gen_fps"] - 10.0) < 1e-6


def test_shown_fps_counts_pictures_in_the_last_second() -> None:
    rt = _bare_stream()
    StreamRuntime._reset_display_clock(rt)
    # A pair 2 ms apart then a long gap read ~170 fps as an EMA of 1/gap.
    for t in (1.0, 1.002, 1.5, 1.502, 2.0):
        fps = StreamRuntime._shown_fps(rt, t)
    assert 3.5 < fps < 4.5


def test_snap_blend_only_smooths_a_still_face() -> None:
    """Blending through a move left the old head and hair on screen for 3–4
    keys: hair trailing the face after a turn."""
    from backend.engine import neutral_keypoints

    rt = _bare_stream()
    rt._status["frame_blend"] = 0.5
    rt._ema_frame = None
    rt._ema_kps = None
    still = neutral_keypoints()
    moved = still.copy()
    moved[:28, 0] += 0.05
    StreamRuntime._blend_display_frame(rt, _img(0), still)
    # Same pose: smoothed half-way (flicker hiding still works).
    out = StreamRuntime._blend_display_frame(rt, _img(200), still)
    assert 90 <= int(np.asarray(out).mean()) <= 110
    # The face moved: show the new key as-is.
    out = StreamRuntime._blend_display_frame(rt, _img(40), moved)
    assert int(np.asarray(out).mean()) == 40
