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


def test_gen_thread_queues_raw_keys_for_the_display_to_blend() -> None:
    """The snap blend used to run between two DiT calls (~17 ms a key)."""
    rt = _bare_stream()
    rt._emit = lambda ev: None  # type: ignore[method-assign]
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
    rt._schedule_next_frame = lambda: None  # type: ignore[method-assign]
    rt._blend_display_frame = (  # type: ignore[method-assign]
        lambda image, kps=None: (_ for _ in ()).throw(AssertionError("blended on gen thread"))
    )
    raw = _img(7)
    StreamRuntime._on_stream_keys(rt, [raw], [None], 0.1, {})
    job = rt._display_queue.get_nowait()
    assert job["keys"][0][0] is raw
    assert job["blend"] is True


def _player(rt: StreamRuntime) -> list[tuple[int, bool]]:
    shown: list[tuple[int, bool]] = []
    rt._pace_display = lambda gap=0.0: None  # type: ignore[method-assign]
    rt._publish_display_frame = (  # type: ignore[method-assign]
        lambda image, kps, key=True: shown.append((int(np.asarray(image).mean()), key))
    )
    return shown


def test_display_blends_keys_and_tweens_from_what_it_showed() -> None:
    rt = _bare_stream()
    rt._status["frame_blend"] = 0.5
    shown = _player(rt)
    job = {"keys": [(_img(100), None)], "prev": None, "count": 1, "blend": True, "epoch": 1}
    StreamRuntime._play_display_job(rt, job)
    # Raw prev says 0, but 100 is on screen: the mid starts from there.
    job = {"keys": [(_img(200), None)], "prev": _img(0), "count": 1, "key_interval": 0.1, "blend": True, "epoch": 1}
    StreamRuntime._play_display_job(rt, job)
    assert shown[0] == (100, True)
    mid, key = shown[1], shown[2]
    # Key blended half way (still face); the mid sits between 100 and it.
    assert key == (150, True)
    assert mid[1] is False and 100 <= mid[0] <= 150


def test_new_stream_does_not_tween_from_the_last_one() -> None:
    rt = _bare_stream()
    shown = _player(rt)
    StreamRuntime._play_display_job(
        rt, {"keys": [(_img(90), None)], "count": 1, "key_interval": 0.1, "blend": True, "epoch": 1}
    )
    StreamRuntime._play_display_job(
        rt, {"keys": [(_img(30), None)], "count": 1, "key_interval": 0.1, "blend": True, "epoch": 2}
    )
    # No mid from 90 and no blend with it: the new stream opens on its key.
    assert shown == [(90, True), (30, True)]


def test_lone_slow_key_is_not_held_back_by_its_mids() -> None:
    from backend.frame_interp import TWEEN_MAX_S

    rt = _bare_stream()
    paced: list[float] = []
    rt._pace_display = lambda gap=0.0: paced.append(gap)  # type: ignore[method-assign]
    rt._publish_display_frame = lambda image, kps, key=True: None  # type: ignore[method-assign]
    # 3 keys/s with up to 3 mids: they used to fill the whole 333 ms gap.
    job = {"keys": [(_img(200), None)], "prev": _img(0), "count": 3, "key_interval": 1 / 3}
    StreamRuntime._play_display_job(rt, job)
    assert len(paced) >= 2
    assert sum(paced) <= TWEEN_MAX_S + 1e-9


def test_call_from_before_a_restart_is_dropped() -> None:
    """A DiT call started before Stop and finished after Start used to land
    as the new stream's first key and clear its in-flight flag."""
    import queue as _queue
    import threading as _threading
    from types import SimpleNamespace

    rt = _bare_stream()
    rt._stream_epoch = 2
    rt._frame_in_flight = True  # the new stream's own call is queued
    rt._gen_queue = _queue.Queue()
    rt._worker_stop = _threading.Event()
    rt._gen_busy = False
    rt._offload_pending = False
    landed: list[int] = []
    rt._on_stream_keys = lambda *a, **k: landed.append(1)  # type: ignore[method-assign]
    rt._note_live_call = lambda *a: None  # type: ignore[method-assign]
    rt._maybe_offload_after_stop = lambda: None  # type: ignore[method-assign]
    rt.engine = SimpleNamespace(
        generate_batch_from_keypoints=lambda *a, **k: ([_img(5)], 0.1),
        last_timings={},
        last_target_keypoints_batch=None,
    )
    kps = np.zeros((37, 4), dtype=np.float32)
    rt._gen_queue.put({"steps": 1, "streaming": True, "epoch": 1, "keypoints": kps})
    rt._gen_queue.put(None)
    StreamRuntime._gen_worker_loop(rt)
    assert landed == []
    assert rt._frame_in_flight is True
    assert rt._streaming is True


def test_stream_status_is_throttled() -> None:
    rt = _bare_stream()
    sent: list[dict] = []
    rt._emit = lambda ev: sent.append(ev)  # type: ignore[method-assign]
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
    for _ in range(10):
        StreamRuntime._emit_stream_status(rt)
    assert len(sent) == 1


def test_a_failed_schedule_does_not_freeze_the_stream(monkeypatch) -> None:
    """On a Timer thread the exception escaped with the in-flight flag set."""
    import threading as _threading

    rt = _bare_stream()
    rt._gen_hold_pending = False
    rt._last_gen_start = 0.0
    rt._status["max_fps"] = 0
    rt.engine = __import__("types").SimpleNamespace(stream_batch_size=1)
    rt._current_keypoints = lambda: (_ for _ in ()).throw(RuntimeError("lab frame broke"))  # type: ignore[method-assign]
    timers: list[float] = []

    class _Timer:
        def __init__(self, delay, fn):
            timers.append(delay)

        def start(self):
            return None

    monkeypatch.setattr(_threading, "Timer", _Timer)
    StreamRuntime._schedule_next_frame(rt)
    assert rt._frame_in_flight is False
    assert timers == [0.25]
    assert rt._streaming is True


def test_call_running_through_a_restart_is_dropped() -> None:
    import queue as _queue
    import threading as _threading
    from types import SimpleNamespace

    rt = _bare_stream()
    rt._stream_epoch = 1
    rt._gen_queue = _queue.Queue()
    rt._worker_stop = _threading.Event()
    rt._gen_busy = False
    rt._offload_pending = False
    landed: list[int] = []
    rt._on_stream_keys = lambda *a, **k: landed.append(1)  # type: ignore[method-assign]
    rt._note_live_call = lambda *a: None  # type: ignore[method-assign]
    rt._maybe_offload_after_stop = lambda: None  # type: ignore[method-assign]

    def generate(*_a, **_k):
        # Stop + Start while this call is on the GPU.
        rt._stream_epoch = 3
        rt._frame_in_flight = True
        return [_img(5)], 0.1

    rt.engine = SimpleNamespace(
        generate_batch_from_keypoints=generate, last_timings={}, last_target_keypoints_batch=None
    )
    kps = np.zeros((37, 4), dtype=np.float32)
    rt._gen_queue.put({"steps": 1, "streaming": True, "epoch": 1, "keypoints": kps})
    rt._gen_queue.put(None)
    StreamRuntime._gen_worker_loop(rt)
    assert landed == []
    assert rt._frame_in_flight is True
