from backend import gpu_monitor
from backend.gpu_monitor import gpu_utilization


def test_no_card_reads_none() -> None:
    assert gpu_utilization("") is None
    assert gpu_utilization("GPU-not-a-real-card", now=1.0) is None


def test_reads_are_cached_per_window(monkeypatch) -> None:
    calls: list[int] = []

    class _Lib:
        def nvmlDeviceGetHandleByUUID(self, uuid, handle):  # noqa: N802
            return 0

        def nvmlDeviceGetUtilizationRates(self, handle, util):  # noqa: N802
            calls.append(1)
            util._obj.gpu = 42
            return 0

    monkeypatch.setattr(gpu_monitor, "_nvml", _Lib())
    monkeypatch.setattr(gpu_monitor, "_cache", {})
    monkeypatch.setattr(gpu_monitor, "_handles", {})
    assert gpu_utilization("GPU-x", now=10.0) == 42
    assert gpu_utilization("GPU-x", now=10.2) == 42
    assert len(calls) == 1
    gpu_utilization("GPU-x", now=10.0 + gpu_monitor.SAMPLE_S)
    assert len(calls) == 2
