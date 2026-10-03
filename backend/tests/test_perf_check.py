"""Performance test: fixed motion loop, timing maths, and the desk start/stop flow."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from backend import perf_check
from backend.engine import neutral_keypoints
from backend.perf_check import MOTION_PERIOD_S, WARMUP_S, PerfRun, motion_pose
from backend.stream import StreamRuntime


def test_motion_moves_the_head_and_loops() -> None:
    base = neutral_keypoints()
    a = motion_pose(base, 0.3)
    b = motion_pose(base, 1.1)
    assert a.shape == base.shape and a.dtype == np.float32
    assert not np.allclose(a[:31, :2], b[:31, :2])
    np.testing.assert_allclose(motion_pose(base, 0.3 + MOTION_PERIOD_S), a, atol=1e-5)
    # Confidence / visibility columns are the still's own.
    np.testing.assert_array_equal(a[:, 2:], base[:, 2:])
    assert np.all(np.abs(a[:, :2]) < 1.0)


def test_motion_leaves_the_base_alone() -> None:
    base = neutral_keypoints()
    before = base.copy()
    motion_pose(base, 2.0)
    np.testing.assert_array_equal(base, before)


def _run_at_20fps(seconds: float = 4.0, hitch_at: float | None = None) -> PerfRun:
    run = PerfRun(seconds, 0.0, label="alone")
    t = 2.0  # model load: the clock waits for the first picture
    first = t
    while t < first + WARMUP_S + seconds:
        run.note_frame(t)
        run.note_call(t, 0.05, 1)
        run.sample(t, gpu=40.0, vram_mb=3000.0, cpu=12.0)
        t += 0.05
        if hitch_at is not None and abs(t - first - WARMUP_S - hitch_at) < 0.025:
            t += 0.5
    return run


def test_result_reads_steady_frame_rate_after_warmup() -> None:
    run = _run_at_20fps()
    res = run.result()
    assert res["label"] == "alone"
    assert res["fps"] == pytest.approx(20.0, abs=0.5)
    assert res["fps_low"] == pytest.approx(20.0, abs=0.5)
    assert res["stalls"] == 0
    assert res["keys_per_s"] == pytest.approx(20.0, abs=0.5)
    assert res["key_ms"] == pytest.approx(50.0)
    assert res["gpu"] == 40 and res["vram_mb"] == 3000 and res["cpu"] == 12
    assert res["seconds"] == pytest.approx(4.0, abs=0.15)


def test_a_hitch_shows_in_the_one_percent_low_and_stalls() -> None:
    res = _run_at_20fps(seconds=8.0, hitch_at=3.0).result()
    assert res["stalls"] == 1
    assert res["fps_low"] < 10.0


def test_done_waits_for_the_first_picture_then_times_out() -> None:
    run = PerfRun(10.0, 0.0)
    assert not run.done(50.0)
    assert run.progress(50.0) == 0.0
    assert run.done(perf_check.START_TIMEOUT_S + 1.0)
    run2 = PerfRun(10.0, 0.0)
    run2.note_frame(5.0)
    assert not run2.done(5.0 + WARMUP_S + 9.0)
    assert run2.done(5.0 + WARMUP_S + 10.0)
    run3 = PerfRun(10.0, 0.0)
    run3.stop.set()
    assert run3.done(0.0)


def _runtime(streaming: bool = False) -> StreamRuntime:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._listeners = []
    rt._status = {"perf_test": {"running": False, "progress": 0.0, "label": "", "results": []}}
    rt._perf = None
    rt._streaming = streaming
    rt._perf_loop = lambda run: None  # type: ignore[method-assign]
    calls: list[str] = []
    rt.calls = calls  # type: ignore[attr-defined]

    def start_stream() -> None:
        calls.append("start")
        rt._streaming = True

    def stop_stream() -> None:
        calls.append("stop")
        rt._streaming = False

    rt.start_stream = start_stream  # type: ignore[method-assign]
    rt.stop_stream = stop_stream  # type: ignore[method-assign]
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
    return rt


def test_start_runs_the_stream_and_finish_stops_it_with_a_result() -> None:
    rt = _runtime()
    st = rt.start_perf_test(5, "")
    assert st["perf_test"]["running"] and st["perf_test"]["label"] == "Run 1"
    run = rt._perf
    assert run is not None and run.started_stream
    with pytest.raises(RuntimeError):
        rt.start_perf_test(5)
    for i in range(80):
        run.note_frame(i * 0.05)
    rt._finish_perf(run)
    state = rt._status["perf_test"]
    assert rt._perf is None and not state["running"]
    assert [r["label"] for r in state["results"]] == ["Run 1"]
    assert rt.calls == ["start", "stop"]


def test_test_on_a_running_stream_leaves_it_running() -> None:
    rt = _runtime(streaming=True)
    rt.start_perf_test(5, "with game")
    run = rt._perf
    for i in range(80):
        run.note_frame(i * 0.05)
    rt._finish_perf(run)
    assert rt.calls == [] and rt._streaming
    assert rt._status["perf_test"]["results"][0]["label"] == "with game"


def test_no_pictures_means_no_result() -> None:
    rt = _runtime()
    rt.start_perf_test(5)
    rt._finish_perf(rt._perf)
    assert rt._status["perf_test"]["results"] == []


def test_current_keypoints_follow_the_test_loop() -> None:
    rt = _runtime(streaming=True)

    class Engine:
        _ref_keypoints = neutral_keypoints()

    rt.engine = Engine()
    rt._perf = PerfRun(10.0, 0.0)
    a = StreamRuntime._current_keypoints(rt)
    assert a is not None and a.shape == (37, 4)
    assert not np.allclose(a, Engine._ref_keypoints)


def test_clear_keeps_results_while_a_test_runs() -> None:
    rt = _runtime()
    rt._status["perf_test"]["results"] = [{"label": "Run 1"}]
    rt._perf = PerfRun(10.0, 0.0)
    rt.clear_perf_results()
    assert rt._status["perf_test"]["results"] == [{"label": "Run 1"}]
    rt._perf = None
    rt.clear_perf_results()
    assert rt._status["perf_test"]["results"] == []
