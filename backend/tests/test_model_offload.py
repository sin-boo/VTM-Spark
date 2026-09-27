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
    rt.status = lambda: dict(rt._status)  # type: ignore[method-assign]
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


def test_model_switch_after_offload_brings_decoders_back(monkeypatch, tmp_path) -> None:
    """Stop stream parks the VAEs in RAM; loading a new DiT must move them back
    before marking the engine GPU-resident, or decode mixes cuda input with cpu weights."""
    import backend.engine as engine_module

    class _Parked:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.moves: list[str] = []

        def to(self, device: torch.device) -> "_Parked":
            self.moves.append(str(device))
            return self

    ckpt = tmp_path / "next.pt"
    ckpt.write_bytes(b"weights")
    engine = StreamEngine(checkpoint=ckpt, device="cpu")
    engine.device = torch.device("cuda")  # descriptor only; moves are recorded
    vae, tiny, latent = _Parked(), _Parked(), _Parked()
    engine.vae = vae
    engine.vae_tiny = tiny
    engine._ref_latent = latent  # type: ignore[assignment]
    engine._gpu_resident = False
    monkeypatch.setattr(
        engine_module,
        "build_keypoint_model",
        lambda _path, _device, dtype=None: (
            torch.nn.Identity(),
            {"use_keypoint_conditioning": True, "image_resolution": 768},
        ),
    )

    engine.load()

    assert engine._gpu_resident is True
    assert vae.moves == ["cuda"]
    assert tiny.moves == ["cuda"]
    assert latent.moves == ["cuda"]


def test_generate_rechecks_gpu_under_lock() -> None:
    """A deferred offload can land between the caller's ensure_gpu and the
    CUDA lock; the locked generate path must move the weights back itself."""
    import threading

    import pytest

    engine = object.__new__(StreamEngine)
    engine._cuda_lock = threading.RLock()
    engine._ready = True
    engine._ref_latent = None
    engine._ref_keypoints = None
    calls: list[str] = []
    engine.ensure_gpu = lambda: calls.append("gpu")  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        StreamEngine._denoise_to_latents_locked(engine, None)  # type: ignore[arg-type]
    assert calls == ["gpu"]


def test_warmup_rechecks_gpu_after_compile_gap() -> None:
    src = inspect.getsource(StreamEngine.warmup)
    after_compile = src.split("self._maybe_compile_model(")[1]
    assert "self.ensure_gpu()" in after_compile


def test_offload_waits_for_warmup_and_warmup_cancels_it() -> None:
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._streaming = False
    rt._gen_busy = False
    rt._offload_pending = True
    rt.status = lambda: {"fast_warming": True}  # type: ignore[method-assign]
    moved: list[bool] = []

    class _Eng:
        def offload_to_cpu(self) -> bool:
            moved.append(True)
            return True

    rt.engine = _Eng()  # type: ignore[assignment]
    StreamRuntime._offload_models(rt)
    assert moved == []
    assert rt._offload_pending is True

    src = inspect.getsource(StreamRuntime._run_fast_warmup_if_needed)
    assert src.index("self._offload_pending = False") < src.index("self.engine.ensure_gpu()")
