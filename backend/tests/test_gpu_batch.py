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
    pick = inspect.getsource(StreamRuntime._pick_auto_batch2)
    assert "_pick_auto_batch2" in warm
    assert "_pick_auto_batch2" in warm.split("self.engine.warmup")[0]
    assert "_restore_eager_model" not in pick
    assert "force=True" not in pick
    toggle = inspect.getsource(StreamRuntime.update_settings)
    batch_chunk = toggle.split('if "batch2" in updates')[1].split("self._emit")[0]
    assert "_restore_eager_model" not in batch_chunk
