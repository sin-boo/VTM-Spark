from backend.stream import should_auto_batch2


def test_auto_batch2_when_gpu_under_90_percent() -> None:
    assert should_auto_batch2(0.0) is True
    assert should_auto_batch2(0.5) is True
    assert should_auto_batch2(0.899) is True


def test_auto_batch2_skips_when_gpu_at_or_over_90_percent() -> None:
    assert should_auto_batch2(0.9) is False
    assert should_auto_batch2(0.99) is False
    assert should_auto_batch2(1.0) is False


def test_auto_batch2_skips_without_gpu_reading() -> None:
    assert should_auto_batch2(None) is False


def test_warmup_picks_batch_before_compile() -> None:
    import inspect

    from backend.stream import StreamRuntime

    warm = inspect.getsource(StreamRuntime._run_fast_warmup_if_needed)
    pick = inspect.getsource(StreamRuntime._pick_batch)
    assert "_pick_batch" in warm
    assert "_pick_batch" in warm.split("self.engine.warmup")[0]
    assert "_restore_eager_model" not in pick
    assert "force=True" not in pick
    toggle = inspect.getsource(StreamRuntime.update_settings)
    batch_chunk = toggle.split("save_ui_session(batch=")[1].split("self._emit")[0]
    assert "_restore_eager_model" not in batch_chunk


def _pick_rt(batch: object, vram: float | None):
    import threading
    from pathlib import Path

    from backend import stream as stream_mod
    from backend.stream import StreamRuntime

    class _Eng:
        stream_batch_size = 1
        checkpoint = Path("VTM-test.pt")
        compile_model = True
        device = None

        def set_stream_batch_size(self, n: int) -> None:
            self.stream_batch_size = n

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._listeners = []
    rt._status = {"batch": batch, "steps": 1, "fast_mode": True, "max_fps": 0, "interpolate": True, "inbetweens": -1}
    rt._streaming = False
    rt._batch_picked = False
    rt._batch_rates = {}
    rt.engine = _Eng()
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
    rt._set_status = lambda **kw: rt._status.update(kw)  # type: ignore[method-assign]
    stream_mod.gpu_used_fraction = lambda engine=None: vram  # type: ignore[assignment]
    return rt


def test_pick_batch_uses_the_setting_or_this_pcs_timings(monkeypatch) -> None:
    from backend import hw_profile
    from backend import stream as stream_mod
    from backend.stream import StreamRuntime

    saved = stream_mod.gpu_used_fraction
    measured: dict[int, float] = {}
    monkeypatch.setattr(hw_profile, "load_rates", lambda key, path=None: dict(measured))
    monkeypatch.setattr(hw_profile, "load_failed", lambda key, path=None: set())
    try:
        # A fixed setting is used as-is, VRAM or not.
        for setting, want in ((3, 3), (1, 1), (9, 4)):
            rt = _pick_rt(setting, 0.95)
            StreamRuntime._pick_batch(rt)
            assert rt.engine.stream_batch_size == want, setting
        # Auto, VRAM nearly full: x1.
        rt = _pick_rt(0, 0.95)
        StreamRuntime._pick_batch(rt)
        assert rt.engine.stream_batch_size == 1
        # Auto, nothing measured yet: start at x1 (warmup tunes from there).
        rt = _pick_rt(0, 0.5)
        StreamRuntime._pick_batch(rt)
        assert rt.engine.stream_batch_size == 1
        # Auto on a PC that measured like a 5060 Ti: x2.
        measured.update({1: 0.102, 2: 0.142})
        rt = _pick_rt(0, 0.5)
        StreamRuntime._pick_batch(rt)
        assert rt.engine.stream_batch_size == 2
        # Picked once per warmup: a second call does not re-plan.
        rt._status["batch"] = 4
        StreamRuntime._pick_batch(rt)
        assert rt.engine.stream_batch_size == 2
    finally:
        stream_mod.gpu_used_fraction = saved


def test_clip_batch() -> None:
    from backend.engine import STREAM_BATCH_MAX, _clip_batch

    assert STREAM_BATCH_MAX == 4
    assert _clip_batch(None) == 0
    assert _clip_batch("3") == 3
    assert _clip_batch(-1) == 0
    assert _clip_batch(12) == 4
