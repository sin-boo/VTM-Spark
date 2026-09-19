import inspect
import os

import torch

from backend.engine import StreamEngine
from backend.paths import configure_torch_compile_cache, torch_compile_cache_dir
from backend.stream import StreamRuntime


def test_compile_cache_points_under_models() -> None:
    configure_torch_compile_cache()
    cache = torch_compile_cache_dir()
    assert cache.is_dir()
    assert "models" in cache.as_posix()
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    assert os.environ.get("TRITON_CACHE_DIR")


def test_cpu_offload_is_noop() -> None:
    engine = StreamEngine(checkpoint="missing.pt", device="cpu")
    engine._ready = True
    engine._gpu_resident = True
    engine.model = torch.nn.Identity()
    assert engine.offload_to_cpu() is False
    engine.ensure_gpu()
    assert engine.model is not None


def test_stop_stream_queues_offload() -> None:
    src = inspect.getsource(StreamRuntime.stop_stream)
    assert "_offload_pending" in src
    assert "_maybe_offload_after_stop" in src
    worker = inspect.getsource(StreamRuntime._gen_worker_loop)
    assert "_maybe_offload_after_stop" in worker
    assert "streaming and not self._streaming" in worker


def test_maybe_offload_calls_engine() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._streaming = False
    rt._gen_busy = False
    rt._offload_pending = True
    rt._fast_warmed = True
    rt._status = {"message": "Stream stopped"}

    def _set(**kwargs: object) -> None:
        rt._status.update(kwargs)

    rt._set_status = _set  # type: ignore[method-assign]
    moved: list[bool] = []

    class _Eng:
        def offload_to_cpu(self) -> bool:
            moved.append(True)
            return True

    rt.engine = _Eng()  # type: ignore[assignment]
    StreamRuntime._maybe_offload_after_stop(rt)
    assert moved == [True]
    assert rt._fast_warmed is False
    assert rt._offload_pending is False
    assert rt._status.get("models_on_gpu") is False


def test_start_tracking_does_not_clear_generate_busy() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._streaming = True
    rt._frame_in_flight = True
    rt._gen_busy = True
    rt._status = {"busy": True, "message": "Streaming"}

    def _set(**kwargs: object) -> None:
        rt._status.update(kwargs)

    rt._set_status = _set  # type: ignore[method-assign]
    StreamRuntime._set_track_status(rt, busy=False, track_busy=False, tracking=True)
    assert rt._status["busy"] is True
    assert rt._status["tracking"] is True
