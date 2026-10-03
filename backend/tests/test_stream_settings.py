from backend.engine import (
    STREAM_DEFAULT_ID_CFG,
    STREAM_DEFAULT_POSE_CFG,
    STREAM_DEFAULT_STEPS,
    STREAM_INBETWEENS,
    STREAM_MAX_GEN_FPS,
    STREAM_TEMPORAL_EMA,
    StreamEngine,
    _clip_blend,
    _clip_cfg,
    _clip_inbetweens,
    _clip_max_fps,
    effective_inbetweens,
    gen_cap,
    gen_hold_s,
    interpolate_on,
    show_fps_max,
)


def test_clip_cfg_bounds() -> None:
    assert _clip_cfg(-1.0) == 0.0
    assert _clip_cfg(9.0) == 6.0
    assert _clip_cfg(2.25) == 2.25


def test_clip_inbetweens_bounds() -> None:
    # Negative = Auto (fill the display at this PC's key rate).
    assert _clip_inbetweens(-1) == -1
    assert _clip_inbetweens(-7) == -1
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
    assert _clip_max_fps(None) == STREAM_MAX_GEN_FPS
    assert _clip_max_fps(-5) == 0
    assert _clip_max_fps("14") == 14
    assert _clip_max_fps(60) == 60
    assert _clip_max_fps(999) == 100


def test_show_fps_follows_cap_above_20() -> None:
    # Auto and low caps keep the 20 fps display; a higher cap speeds it up.
    assert show_fps_max(0) == 20.0
    assert show_fps_max(12) == 20.0
    assert show_fps_max(60) == 60.0
    assert show_fps_max(999) == 100.0


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


def test_auto_cap_fills_the_display_with_mids() -> None:
    # Auto (0): keys + in-betweens = the 20 fps the display shows.
    assert gen_cap(0, True, 1) == 10.0
    assert gen_cap(0, True, 3) == 5.0
    assert abs(gen_cap(0, True, 2) - 20.0 / 3.0) < 1e-9
    # Interpolate off: keys only, still no point past the display.
    assert gen_cap(0, False, 3) == 20.0
    # A manual cap wins.
    assert gen_cap(7, True, 1) == 7.0
    # Fractional Auto caps pace exactly (no rounding 6.67 up to 7).
    assert abs(gen_hold_s(20.0 / 3.0, 10.0, 10.0) - 0.15) < 1e-9


def test_auto_show_fps_matches_display() -> None:
    from backend.engine import STREAM_MAX_GEN_FPS_LIMIT, STREAM_SHOW_FPS
    from backend.frame_interp import SHOW_FPS_MAX

    assert STREAM_SHOW_FPS == SHOW_FPS_MAX
    assert STREAM_MAX_GEN_FPS_LIMIT >= SHOW_FPS_MAX


def test_snap_alpha_eases_off_with_motion() -> None:
    from backend.engine import snap_alpha

    assert snap_alpha(0.58, 0.0) == 0.58
    assert snap_alpha(0.58, 0.5) == 1.0
    mid = snap_alpha(0.58, 0.015)
    assert 0.58 < mid < 1.0


def test_auto_inbetweens_ask_for_the_max_and_aim_keys_at_ten() -> None:
    from backend.engine import STREAM_MAX_INBETWEENS

    assert effective_inbetweens(True, -1) == STREAM_MAX_INBETWEENS
    assert effective_inbetweens(False, -1) == 0
    # Real keys up to 10/s, mids fill the rest of the 20 fps display.
    assert gen_cap(0, True, -1) == 10.0
    assert gen_cap(0, False, -1) == 20.0


def test_out_of_the_box_speed() -> None:
    """Out of the box: 30 keys/s cap, one in-between, batch 3. Auto held keys
    to 10/s, and a game beside it pushed the in-betweens out: ~9 fps."""
    from backend.engine import STREAM_BATCH_DEFAULT

    assert STREAM_MAX_GEN_FPS == 30
    assert STREAM_INBETWEENS == 1
    assert STREAM_BATCH_DEFAULT == 3
    assert gen_cap(STREAM_MAX_GEN_FPS, True, STREAM_INBETWEENS) == 30.0
