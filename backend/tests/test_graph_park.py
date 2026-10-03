"""The fp32 DiT and SD-VAE wait in RAM while keys run on the frame graph."""

import inspect
import re

import numpy as np
import pytest
import torch

import backend.engine as E
from backend.engine import StreamEngine


class _Mod:
    """Stands in for a module: records where it was moved."""

    def __init__(self, name: str = "m") -> None:
        self.name = name
        self.device = torch.device("cuda")
        self.moves: list[str] = []

    def to(self, device) -> "_Mod":
        self.device = torch.device(device)
        self.moves.append(self.device.type)
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


class _Compiled:
    def __init__(self, orig: _Mod) -> None:
        self._orig_mod = orig


def _engine(monkeypatch) -> StreamEngine:
    engine = StreamEngine(checkpoint="missing.pt", device="cpu", fast_mode=True)
    engine.device = torch.device("cuda")  # descriptor only; moves are recorded
    engine.model = _Mod("dit")
    engine._eager_model = engine.model
    engine.vae = _Mod("vae")
    engine._gpu_resident = True
    engine._ready = True
    cleared: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    engine.cleared = cleared  # type: ignore[attr-defined]
    return engine


def test_park_moves_dit_and_vae_to_ram_and_frees_vram(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    dit, vae = engine.model, engine.vae
    engine._park_idle()
    assert engine._parked == {"dit", "vae"}
    assert dit.device.type == "cpu" and vae.device.type == "cpu"
    assert engine.model is dit and engine._eager_model is dit
    assert engine.cleared == [True]
    engine._park_idle()  # already parked: no second move, no second empty_cache
    assert dit.moves == ["cpu"] and engine.cleared == [True]


def test_unpark_brings_back_only_what_is_asked(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    dit, vae = engine.model, engine.vae
    engine._park_idle()
    engine._unpark("vae")
    assert vae.device.type == "cuda" and dit.device.type == "cpu"
    assert engine._parked == {"dit"}
    engine._unpark()
    assert dit.device.type == "cuda" and engine._parked == set()
    engine._unpark()  # nothing parked: no move
    assert dit.moves == ["cpu", "cuda"]


def test_park_drops_a_compile_wrapper_first(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    eager = engine.model
    engine.model = _Compiled(eager)
    engine._model_compiled = True
    engine._park_idle()
    assert engine.model is eager and engine._eager_model is eager
    assert eager.device.type == "cpu"


def test_unpark_moves_the_eager_model_under_a_later_compile_wrapper(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    eager = engine.model
    engine._park_idle()
    wrapper = _Compiled(eager)
    engine.model = wrapper  # _maybe_compile_model wrapped it while parked
    engine._unpark("dit")
    assert eager.device.type == "cuda"
    assert engine.model is wrapper and engine._eager_model is eager


def test_park_is_a_noop_on_cpu_or_after_offload(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    engine._gpu_resident = False
    engine._park_idle()
    assert engine._parked == set()
    engine._gpu_resident = True
    engine.device = torch.device("cpu")
    engine._park_idle()
    assert engine._parked == set() and engine.model.moves == []


def test_park_never_waits_for_a_side_stream_decode(monkeypatch) -> None:
    """decode_latents_to_images holds _decode_lock (and may wait for _cuda_lock to
    unpark the VAE): parking skips this key instead of deadlocking."""
    engine = _engine(monkeypatch)
    assert engine._decode_lock.acquire(blocking=False)
    try:
        engine._park_idle()
    finally:
        engine._decode_lock.release()
    assert engine._parked == set() and engine.vae.moves == []
    engine._park_idle()
    assert engine._parked == {"dit", "vae"}


def test_failed_unpark_stays_parked_and_raises(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    engine._park_idle()

    def oom(device):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    engine.model.to = oom  # type: ignore[method-assign]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine._unpark("dit")
    assert "dit" in engine._parked  # the next eager call retries the move


def test_sd_decode_unparks_the_vae(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    engine.fast_mode = False
    engine._park_idle()
    seen: list[str] = []
    monkeypatch.setattr(E, "decode_sd_vae", lambda vae, _lat: seen.append(vae.device.type) or np.zeros(1))
    engine._decode_latents(torch.zeros(1))
    assert seen == ["cuda"]
    assert engine._parked == {"dit"}  # decode does not need the DiT


def test_ensure_gpu_leaves_graph_only_weights_in_ram(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    engine._gpu_resident = False
    for m in (engine.model, engine.vae):
        m.device = torch.device("cpu")
    monkeypatch.setattr(engine, "_graph_ok", lambda *a: True)
    engine.ensure_gpu()
    assert engine._gpu_resident is True
    assert engine._parked == {"dit", "vae"}
    assert engine.model.moves == [] and engine.vae.moves == []

    engine._gpu_resident = False
    engine._parked.clear()
    monkeypatch.setattr(engine, "_graph_ok", lambda *a: False)
    engine.ensure_gpu()
    assert engine._parked == set()
    assert engine.model.device.type == "cuda" and engine.vae.device.type == "cuda"


def test_offload_and_model_switch_reset_the_parked_set(monkeypatch, tmp_path) -> None:
    engine = _engine(monkeypatch)
    monkeypatch.setattr(E, "_clear_cuda_errors", lambda: None)
    engine._park_idle()
    assert engine.offload_to_cpu() is True
    assert engine._parked == set()

    engine._park_idle()  # not resident: nothing
    engine._gpu_resident = True
    engine._park_idle()
    engine._release_dit_weights()
    assert engine._parked == {"vae"}
    ckpt = tmp_path / "next.pt"
    ckpt.write_bytes(b"w")
    engine.checkpoint = ckpt
    monkeypatch.setattr(
        E,
        "build_keypoint_model",
        lambda _p, _d, dtype=None: (_Mod("dit2"), {"use_keypoint_conditioning": True}),
    )
    engine.load()
    assert engine._parked == set()
    assert engine.vae.device.type == "cuda"


class _FakeGraph:
    compiled = False

    def __init__(self, mode: str, dtype: str) -> None:
        self.mode, self.dtype_name, self.decoder_name = mode, dtype, "fake"

    def prepare(self, bsz: int) -> None:
        pass

    def run(self, **_kw):
        return torch.zeros(1, 4, 2, 2), np.zeros((1, 2, 2, 3), np.uint8), True


def test_graph_build_parks_before_copying_and_keys_repark(monkeypatch, tmp_path) -> None:
    engine = _engine(monkeypatch)
    monkeypatch.setattr(E, "fast_decoder_path", lambda: tmp_path / "none.pt")
    monkeypatch.setattr(engine, "_ensure_tiny_vae", lambda: True)
    parked_at_build: list[set[str]] = []

    def build(mode: str, dt: str) -> _FakeGraph:
        parked_at_build.append(set(engine._parked))
        return _FakeGraph(mode, dt)

    monkeypatch.setattr(engine, "_build_graph_frame", build)
    kps = np.zeros((1, 37, 4), np.float32)
    assert engine._run_graph_frame(kps, None, None, 0.0) is not None
    assert parked_at_build == [{"dit", "vae"}]  # copied from RAM, not next to it in VRAM

    engine._unpark("vae")  # a new reference was encoded mid-stream
    assert engine._run_graph_frame(kps, None, None, 0.0) is not None
    assert engine._parked == {"dit", "vae"}
    assert len(parked_at_build) == 1


def test_every_eager_use_unparks_first() -> None:
    """Any method that runs the fp32 DiT or the SD-VAE must fetch it from RAM
    before the call, or a parked weight meets a CUDA input."""
    src = inspect.getsource(StreamEngine)
    uses = {
        "denoise_keypoint(": '_unpark("dit")',
        "encode_reference(": '_unpark("vae")',
        "decode_sd_vae(": '_unpark("vae")',
    }
    methods = re.split(r"\n    def ", src)
    checked = 0
    for body in methods:
        for call, guard in uses.items():
            at = body.find(call)
            if at < 0:
                continue
            checked += 1
            name = body.split("(", 1)[0]
            assert guard in body[:at], f"{name} calls {call} without {guard} first"
    assert checked >= 5  # verify_compile, warmup, denoise, set_reference, decode


# --- GraphedFrame: reference copied only when it changes -----------------------


def _frame(use_face: bool):
    from backend.graph_frame import GraphedFrame

    gf = object.__new__(GraphedFrame)
    gf.face_size = 4
    s = {
        "ref": torch.zeros(2, 4, 8, 8),
        "face": torch.zeros(2, 4, 4, 4) if use_face else None,
    }
    return gf, s


def test_graph_skips_the_reference_copy_when_unchanged() -> None:
    gf, s = _frame(use_face=True)
    ref = torch.ones(1, 4, 8, 8)
    gf._fill_ref(s, ref, None)
    assert torch.equal(s["ref"], ref.expand(2, -1, -1, -1))
    assert float(s["face"].min()) == 1.0  # face resized from the whole ref
    s["ref"].zero_()
    s["face"].zero_()
    gf._fill_ref(s, ref, None)  # same tensor, same version: no copy
    assert float(s["ref"].abs().max()) == 0.0 and float(s["face"].abs().max()) == 0.0


def test_graph_copies_a_new_or_rewritten_reference() -> None:
    gf, s = _frame(use_face=True)
    ref = torch.ones(1, 4, 8, 8)
    gf._fill_ref(s, ref, None)
    new = torch.full((1, 4, 8, 8), 2.0)  # a new character
    gf._fill_ref(s, new, None)
    assert float(s["ref"].min()) == 2.0
    new.add_(1.0)  # written in place: the version counter moves
    gf._fill_ref(s, new, None)
    assert float(s["ref"].min()) == 3.0
    face = torch.full((1, 4, 4, 4), 5.0)  # a face crop arrives
    gf._fill_ref(s, new, face)
    assert float(s["face"].min()) == 5.0
    assert float(s["ref"].min()) == 3.0


def test_graph_reference_from_inference_mode_is_still_tracked() -> None:
    gf, s = _frame(use_face=False)
    with torch.inference_mode():
        a = torch.ones(1, 4, 8, 8)
        b = torch.full((1, 4, 8, 8), 4.0)
        gf._fill_ref(s, a, None)
        s["ref"].zero_()
        gf._fill_ref(s, a, None)
        assert float(s["ref"].abs().max()) == 0.0
        gf._fill_ref(s, b, None)
        assert float(s["ref"].min()) == 4.0
