import numpy as np
import pytest
import torch

from backend.graph_frame import PoseMap, keypoint_deltas_t
from backend.hw_profile import profile_key
from utils.keypoints import keypoint_deltas
from utils.pose_map import rasterize_pose_maps


def _random_keypoints(bsz: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    kps = np.zeros((bsz, 37, 4), np.float32)
    kps[..., :2] = rng.uniform(-0.9, 0.9, (bsz, 37, 2))
    kps[..., 2] = rng.uniform(0.0, 1.0, (bsz, 37))
    kps[..., 3] = (rng.uniform(0.0, 1.0, (bsz, 37)) > 0.2).astype(np.float32)
    return torch.from_numpy(kps)


def test_pose_map_matches_the_rasterizer() -> None:
    kps = _random_keypoints(3)
    hair = torch.rand(3, 3, 96, 96)
    want = rasterize_pose_maps(kps, 96, 96, sigma=1.5, hair_maps=hair)
    got = PoseMap(96)(kps, hair)
    assert got.shape == want.shape
    assert float((got - want).abs().max()) < 1e-3


def test_pose_map_skips_hidden_points_and_bones() -> None:
    kps = _random_keypoints(1, seed=1)
    kps[..., 3] = 0.0
    out = PoseMap(96)(kps, torch.zeros(1, 3, 96, 96))
    assert float(out.abs().max()) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_keypoint_deltas_match_numpy(seed: int) -> None:
    target = _random_keypoints(2, seed)
    ref = _random_keypoints(1, seed + 10)
    got = keypoint_deltas_t(target, ref.expand(2, -1, -1)).numpy()
    for i in range(2):
        np.testing.assert_allclose(got[i], keypoint_deltas(target[i].numpy(), ref[0].numpy()), atol=1e-6)


def test_graph_runs_keep_their_own_hw_profile() -> None:
    eager = profile_key("RTX", "VTM", 1, True)
    graph = profile_key("RTX", "VTM", 1, True, graph="ultra")
    assert graph != profile_key("RTX", "VTM", 1, True, graph="normal")
    assert graph != eager
    assert graph.startswith(eager)


def _frame_body(decoded: float = 0.0):
    """GraphedFrame._body on CPU with a zero DiT step and a constant decoder."""
    from backend.graph_frame import GraphedFrame
    from utils.keypoints import NUM_POSE_CHANNELS

    n = 8
    gf = object.__new__(GraphedFrame)
    gf.dtype, gf.pose_channels = torch.float32, NUM_POSE_CHANNELS
    gf.pose = PoseMap(n)
    gf.dit = lambda x, *a, **kw: torch.zeros_like(x)
    gf.dec = lambda z: torch.full((z.shape[0], 3, 2 * n, 2 * n), decoded)
    s = {
        "kps_t": torch.zeros(1, 37, 4),
        "kps_r": torch.zeros(1, 37, 4),
        "hair": torch.zeros(1, 3, n, n),
        "ref": torch.zeros(1, 4, n, n),
        "face": None,
        "last": torch.zeros(1, 4, n, n),
        "t0": torch.zeros(1),
        "noise": torch.ones(1, 4, n, n),
    }
    return gf, s


def test_graph_flags_nonfinite_frames() -> None:
    gf, s = _frame_body()
    gf._body(s)
    assert not bool(s["bad_out"])
    s["last"].fill_(float("nan"))
    s["t0"].fill_(0.5)  # hold-last start reads the NaN latent
    gf._body(s)
    assert bool(s["bad_out"])
    gf, s = _frame_body(decoded=float("inf"))  # decoder overflow alone
    gf._body(s)
    assert bool(s["bad_out"])


def test_fresh_frame_ignores_a_stale_nan_hold_latent() -> None:
    gf, s = _frame_body()
    s["last"].fill_(float("nan"))
    gf._body(s)  # t0 = 0
    assert not bool(s["bad_out"])
    assert torch.equal(s["lat_out"], s["noise"])


def _engine(monkeypatch, tmp_path, *, decoder: bool):
    import backend.engine as E

    path = tmp_path / "decoder" / E.STREAM_FAST_DECODER_NAME
    if decoder:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    monkeypatch.setattr(E, "fast_decoder_path", lambda: path)
    return E.StreamEngine(checkpoint="missing.pt", device="cpu", fast_mode=True)


def test_ultra_is_the_graph_mode_with_no_user_toggle(monkeypatch, tmp_path) -> None:
    engine = _engine(monkeypatch, tmp_path, decoder=True)
    assert not hasattr(engine, "set_speed_mode")
    assert not hasattr(engine, "speed_mode")
    assert engine.ultra_available is True
    assert engine.graph_mode == "ultra"
    assert engine.active_speed_mode == "eager"  # CPU: keys never run the graph
    monkeypatch.setattr(engine, "_graph_ok", lambda *a: True)
    assert engine.active_speed_mode == "ultra"


def test_graph_falls_back_to_tinyvae_without_the_decoder(monkeypatch, tmp_path) -> None:
    engine = _engine(monkeypatch, tmp_path, decoder=False)
    monkeypatch.setattr(engine, "_graph_ok", lambda *a: True)
    assert engine.ultra_available is False
    assert engine.graph_mode == "normal"


def test_fast_decoder_kill_switch_and_failure_run_normal(monkeypatch, tmp_path) -> None:
    import backend.engine as E

    engine = _engine(monkeypatch, tmp_path, decoder=True)
    monkeypatch.setattr(engine, "_graph_ok", lambda *a: True)
    monkeypatch.setattr(E, "STREAM_FAST_DECODER", False)  # VTM_FAST_DECODER=0
    assert engine.graph_mode == "normal"
    monkeypatch.setattr(E, "STREAM_FAST_DECODER", True)
    engine._ultra_failed = True
    assert engine.graph_mode == "normal"


@pytest.mark.parametrize(
    ("cap", "override", "want"),
    [
        ((12, 0), "", "fp16"),  # Blackwell
        ((7, 0), "", "fp16"),  # Volta: first with fast fp16
        ((7, 5), "", "fp16"),
        ((6, 1), "", "fp32"),  # Pascal
        ((5, 2), "", "fp32"),
        (None, "", "fp32"),
        ((12, 0), "fp32", "fp32"),
        ((6, 1), "fp16", "fp16"),
        ((6, 1), "bf16", "fp32"),  # unknown override -> automatic
    ],
)
def test_graph_dtype_by_compute_capability(cap, override, want) -> None:
    from backend.engine import graph_dtype_name

    assert graph_dtype_name(cap, override) == want


def test_engine_graph_dtype_follows_override_and_fp32_marks(monkeypatch, tmp_path) -> None:
    import backend.engine as E

    engine = _engine(monkeypatch, tmp_path, decoder=True)
    assert engine.graph_dtype("ultra") == "fp32"  # no CUDA device to read
    monkeypatch.setattr(E, "STREAM_GRAPH_DTYPE", "fp16")
    assert engine.graph_dtype("ultra") == "fp16"
    engine._graph_fp32_modes.add("ultra")
    assert engine.graph_dtype("ultra") == "fp32"
    assert engine.graph_dtype("normal") == "fp16"


def test_nonfinite_frame_plan() -> None:
    from backend.engine import nonfinite_frame_plan

    assert nonfinite_frame_plan("fp16", "ultra") == "fp32"
    assert nonfinite_frame_plan("fp16", "normal") == "fp32"
    assert nonfinite_frame_plan("fp32", "ultra") == "normal"
    assert nonfinite_frame_plan("fp32", "normal") == "eager"


class _FakeGraph:
    compiled = False

    def __init__(self, mode: str, dtype: str, finite: bool) -> None:
        self.mode, self.dtype_name, self.decoder_name = mode, dtype, "fake"
        self.finite = finite
        self.runs = 0

    def prepare(self, bsz: int) -> None:
        pass

    def run(self, **_kw):
        self.runs += 1
        return torch.zeros(1, 4, 2, 2), np.zeros((1, 2, 2, 3), np.uint8), self.finite


def _graph_engine(monkeypatch, tmp_path, finite: dict[str, bool], *, dtype: str = "fp16"):
    """CPU engine whose frame graphs are fakes returning ``finite["mode/dtype"]``,
    else ``finite[dtype]``."""
    import backend.engine as E

    engine = _engine(monkeypatch, tmp_path, decoder=True)
    monkeypatch.setattr(E, "STREAM_GRAPH_DTYPE", dtype)
    monkeypatch.setattr(E, "_clear_cuda_errors", lambda: None)
    engine.model = object()
    monkeypatch.setattr(engine, "_ensure_tiny_vae", lambda: True)
    built: list[tuple[str, str]] = []

    def build(mode: str, dt: str) -> _FakeGraph:
        built.append((mode, dt))
        return _FakeGraph(mode, dt, finite.get(f"{mode}/{dt}", finite.get(dt)))

    monkeypatch.setattr(engine, "_build_graph_frame", build)
    return engine, built


def _key(engine):
    return engine._run_graph_frame(np.zeros((1, 37, 4), np.float32), None, None, 0.0)


def test_finite_frames_reuse_the_ultra_graph(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})
    assert _key(engine) is not None
    assert _key(engine) is not None
    assert built == [("ultra", "fp16")]
    assert engine._graph_frames["ultra"].runs == 2
    assert "graph:fake/fp16" in engine.fast_status


def test_nonfinite_fp16_frame_rebuilds_that_graph_in_fp32(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": False, "fp32": True})
    assert _key(engine) is not None  # the same key, re-run in fp32
    assert built == [("ultra", "fp16"), ("ultra", "fp32")]
    assert engine._graph_failed is False
    assert engine.graph_dtype("ultra") == "fp32"
    assert engine.graph_dtype("normal") == "fp16"
    assert "graph:fake/fp32" in engine.fast_status
    assert _key(engine) is not None
    assert len(built) == 2


def test_broken_fast_decoder_hands_over_to_normal(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(
        monkeypatch, tmp_path, {"ultra/fp16": False, "ultra/fp32": False, "normal/fp16": True}
    )
    assert _key(engine) is not None  # the same key, on the TinyVAE graph
    assert built == [("ultra", "fp16"), ("ultra", "fp32"), ("normal", "fp16")]
    assert engine._graph_failed is False
    assert engine.ultra_available is False
    assert engine.graph_mode == "normal"


def test_nonfinite_everywhere_falls_back_to_eager(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": False, "fp32": False})
    assert _key(engine) is None  # the caller runs this key eager
    assert built == [("ultra", "fp16"), ("ultra", "fp32"), ("normal", "fp16"), ("normal", "fp32")]
    assert engine._graph_failed is True
    assert engine._graph_frames == {}
    assert "graph-fail" in engine.fast_status


def test_nonfinite_frames_on_an_fp32_gpu_skip_the_fp16_retry(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp32": False}, dtype="fp32")
    assert _key(engine) is None
    assert built == [("ultra", "fp32"), ("normal", "fp32")]
    assert engine._graph_failed is True


def test_session_ignores_the_old_speed_mode(tmp_path, monkeypatch) -> None:
    import json

    from backend.ui_session import load_ui_session, save_ui_session, session_path

    monkeypatch.setattr("backend.ui_session.models_root", lambda: tmp_path)
    session_path().write_text(json.dumps({"speed_mode": "normal", "steps": 1}), encoding="utf-8")
    assert "speed_mode" not in load_ui_session()
    save_ui_session(steps=1, speed_mode="normal")
    assert "speed_mode" not in json.loads(session_path().read_text(encoding="utf-8"))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # FastAPI on_event at import
def test_speed_mode_is_not_a_setting() -> None:
    from backend.api import SettingsBody

    assert "speed_mode" not in SettingsBody.model_fields
    assert SettingsBody(speed_mode="normal").model_dump(exclude_none=True) == {}


def _status_rt(engine):
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = engine
    return rt


def test_status_reports_the_mode_read_only() -> None:
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    eng = SimpleNamespace(active_speed_mode="ultra", ultra_available=True)
    fields = StreamRuntime._compile_status_fields(_status_rt(eng))
    assert fields["speed_mode_active"] == "ultra"
    assert fields["ultra_available"] is True
    assert "speed_mode" not in fields
    fields = StreamRuntime._compile_status_fields(_status_rt(SimpleNamespace()))
    assert fields["speed_mode_active"] == "eager"
    assert fields["ultra_available"] is False


def test_hw_profile_key_follows_the_graph_mode_and_dtype() -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    dtypes = {"ultra": "fp16"}
    eng = SimpleNamespace(
        device=None,
        checkpoint=Path("VTM-test.pt"),
        compile_model=True,
        graph_mode="ultra",
        _graph_ok=lambda steps=None: True,
        graph_dtype=lambda mode: dtypes[mode],
    )
    rt = _status_rt(eng)
    rt.status = lambda: {"steps": 1, "fast_mode": True}
    assert StreamRuntime._hw_key(rt).endswith("|graph-ultra")
    dtypes["ultra"] = "fp32"
    assert StreamRuntime._hw_key(rt).endswith("|graph-ultra-fp32")
    eng._graph_ok = lambda steps=None: False
    assert "graph-" not in StreamRuntime._hw_key(rt)


def test_finite_stream_inputs_hide_bad_points_and_hair() -> None:
    from backend.engine import finite_stream_inputs

    kps = np.ones((1, 37, 4), np.float32)
    hair = np.ones((1, 3, 4, 4), np.float32)
    same_k, same_h = finite_stream_inputs(kps, hair)
    assert np.array_equal(same_k, kps) and np.array_equal(same_h, hair)
    kps[0, 5, 0] = np.nan
    hair[0, 1, 2, 2] = np.inf
    k, h = finite_stream_inputs(kps, hair)
    assert np.isfinite(k).all() and np.isfinite(h).all()
    assert np.array_equal(k[0, 5], np.zeros(4))  # hidden, not moved
    assert np.array_equal(k[0, 6], np.ones(4))
    assert h[0, 1, 2, 2] == 0.0
    _, ht = finite_stream_inputs(kps, torch.full((1, 3, 2, 2), float("nan")))
    assert torch.isfinite(ht).all()


def test_bad_tracker_frame_does_not_downgrade_the_graph(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})
    seen = {}
    run = _FakeGraph.run

    def spy(self, **kw):
        seen.update(kw)
        return run(self, **kw)

    monkeypatch.setattr(_FakeGraph, "run", spy)
    kps = np.zeros((1, 37, 4), np.float32)
    kps[0, 3, :2] = np.nan
    assert engine._run_graph_frame(kps, None, None, 0.0) is not None
    assert np.isfinite(seen["kps_target"]).all()
    assert built == [("ultra", "fp16")]
    assert engine.ultra_available and not engine._graph_fp32_modes


def test_capture_oom_is_raised_for_the_batch_tuner(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})

    def oom(self, bsz):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(_FakeGraph, "prepare", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine._ensure_graph_frame(2)
    assert engine.ultra_available and not engine._graph_failed  # nothing sticky
    assert engine._graph_frames == {}


def test_capture_oom_at_batch_1_runs_eager_instead_of_failing_every_key(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})

    def oom(self, **_kw):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(_FakeGraph, "run", oom)
    assert _key(engine) is None  # this key runs eager, nothing raised
    assert engine._graph_failed is True
    assert _key(engine) is None
    assert built == [("ultra", "fp16")]  # no rebuild on every later key


def test_build_oom_at_batch_1_runs_eager(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})

    def oom(mode: str, dt: str):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(engine, "_build_graph_frame", oom)
    assert engine._ensure_graph_frame(1) is None
    assert engine._graph_failed is True


def test_compile_failure_rebuilds_uncompiled_before_blaming_ultra(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})
    monkeypatch.setattr(
        engine, "_graph_compile_mode", lambda: None if engine._graph_skip_compile else "default"
    )
    calls = {"n": 0}

    def prepare(self, bsz):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("triton compile boom")

    monkeypatch.setattr(_FakeGraph, "prepare", prepare)
    assert engine._ensure_graph_frame(1) is not None
    assert built == [("ultra", "fp16"), ("ultra", "fp16")]
    assert engine._graph_skip_compile is True
    assert engine.ultra_available and not engine._graph_failed
    engine.set_compile_model(not engine.compile_model)
    assert engine._graph_skip_compile is False


def test_graph_face_buffer_follows_the_checkpoint_face_size() -> None:
    from backend.graph_frame import GraphedFrame

    gf = object.__new__(GraphedFrame)
    gf.face_size = 16
    s = {"face": torch.full((2, 4, 16, 16), 7.0)}
    gf._fill_face(s, torch.ones(1, 4, 96, 96), torch.full((1, 4, 32, 32), 3.0))
    assert torch.allclose(s["face"], torch.full_like(s["face"], 3.0))
    # No face crop: the whole ref resized (as the DiT does), not the last face kept.
    gf._fill_face(s, torch.ones(1, 4, 96, 96), None)
    assert torch.allclose(s["face"], torch.ones_like(s["face"]))
    s["face"] = None
    gf._fill_face(s, torch.ones(1, 4, 96, 96), None)  # model without face tokens


def test_ultra_graph_runs_without_the_tinyvae(monkeypatch, tmp_path) -> None:
    import backend.engine as E

    engine = _engine(monkeypatch, tmp_path, decoder=True)
    monkeypatch.setattr(engine, "device", torch.device("cuda"))
    monkeypatch.setattr(E, "STREAM_FAST_DISABLE_CFG", True)
    engine._tiny_vae_failed = True  # offline: the TinyVAE download failed
    assert engine._graph_ok(1)
    engine._ultra_failed = True  # Normal needs the TinyVAE: eager
    assert not engine._graph_ok(1)


def test_no_graph_rebuild_once_the_graph_is_off(monkeypatch, tmp_path) -> None:
    engine, built = _graph_engine(monkeypatch, tmp_path, {"fp16": True})
    engine._graph_failed = True
    assert engine._ensure_graph_frame(1) is None
    assert built == []


def test_compile_toggle_retries_ultra_and_the_graph(monkeypatch, tmp_path) -> None:
    engine = _engine(monkeypatch, tmp_path, decoder=True)
    engine._ultra_failed = True
    engine._graph_failed = True
    engine.set_compile_model(not engine.compile_model)
    assert engine.ultra_available and not engine._graph_failed
