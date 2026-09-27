from backend.engine import (
    STREAM_DEFAULT_ID_CFG,
    STREAM_DEFAULT_POSE_CFG,
    STREAM_DEFAULT_STEPS,
    STREAM_INBETWEENS,
    STREAM_TEMPORAL_EMA,
    StreamEngine,
    _clip_blend,
    _clip_cfg,
    _clip_inbetweens,
    _clip_max_fps,
    effective_inbetweens,
    gen_hold_s,
    interpolate_on,
)


def test_clip_cfg_bounds() -> None:
    assert _clip_cfg(-1.0) == 0.0
    assert _clip_cfg(9.0) == 6.0
    assert _clip_cfg(2.25) == 2.25


def test_clip_inbetweens_bounds() -> None:
    assert _clip_inbetweens(-1) == 0
    assert _clip_inbetweens(9) == 3
    assert _clip_inbetweens("2") == 2
    assert _clip_inbetweens("nope") == STREAM_INBETWEENS


def test_interpolate_toggle_wins_over_count() -> None:
    assert interpolate_on(None) is True
    assert interpolate_on(False) is False
    assert effective_inbetweens(False, 2) == 0
    assert effective_inbetweens(True, 2) == 2
    assert effective_inbetweens(None, 1) == 1


def test_clip_blend_bounds() -> None:
    assert _clip_blend(-1.0) == 0.05
    assert _clip_blend(2.0) == 1.0
    assert _clip_blend(STREAM_TEMPORAL_EMA) == STREAM_TEMPORAL_EMA


def test_fast_one_and_two_step_use_joint_forward() -> None:
    engine = StreamEngine(checkpoint="missing.pt", device="cpu", fast_mode=True)
    engine.set_guidance(pose_cfg=2.5, id_cfg=0.7)
    for n in (1, 2):
        steps, pose_cfg, id_cfg = engine._resolve_generate_settings(n)
        assert steps == n
        assert pose_cfg == 1.0
        assert id_cfg == 1.0


def test_guidance_defaults_match_stream() -> None:
    engine = StreamEngine(checkpoint="missing.pt", device="cpu", fast_mode=True)
    steps, pose_cfg, id_cfg = engine._resolve_generate_settings(None)
    assert steps == STREAM_DEFAULT_STEPS
    assert steps == 1
    assert pose_cfg == STREAM_DEFAULT_POSE_CFG
    assert id_cfg == STREAM_DEFAULT_ID_CFG
    assert pose_cfg == 1.0
    assert id_cfg == 1.0


def test_compile_toggle_off_stays_idle() -> None:
    engine = StreamEngine(
        checkpoint="missing.pt",
        device="cpu",
        fast_mode=True,
        compile_model=False,
    )
    assert engine.compile_model is False
    assert engine.compile_status == "off"
    assert "off" in engine.compile_detail.lower()


def test_compile_toggle_on_cpu_skips() -> None:
    engine = StreamEngine(
        checkpoint="missing.pt",
        device="cpu",
        fast_mode=True,
        compile_model=True,
    )
    assert engine.compile_model is True
    assert engine.compile_status == "skip"
    engine.set_compile_model(False)
    assert engine.compile_model is False
    assert engine.compile_status == "off"
    engine.set_compile_model(True)
    assert engine.compile_status == "skip"


def test_max_fps_clip() -> None:
    assert _clip_max_fps(None) == 0
    assert _clip_max_fps(-5) == 0
    assert _clip_max_fps("24") == 24
    assert _clip_max_fps(999) == 60


def test_gen_hold_keeps_keys_under_cap() -> None:
    # Off, or no key yet: never hold.
    assert gen_hold_s(0, 10.0, 10.01) == 0.0
    assert gen_hold_s(10, 0.0, 10.01) == 0.0
    # 10 keys/s: a key 10 ms after the last waits the other 90 ms.
    assert abs(gen_hold_s(10, 10.0, 10.01) - 0.09) < 1e-9
    # Already late: go now.
    assert gen_hold_s(10, 10.0, 10.5) == 0.0
    # Batch x2 makes two keys per call, so it gets twice the interval.
    assert abs(gen_hold_s(10, 10.0, 10.01, batch=2) - 0.19) < 1e-9
