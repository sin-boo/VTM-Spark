"""Pose keys: freeze, persist, and apply keyforms."""

from __future__ import annotations

import numpy as np

from backend.engine import apply_overlay_drag_to_rest, neutral_keypoints
from backend.live_retarget import SemanticControls
from backend.pose_keys import (
    apply_keys,
    key_weight,
    load_keys,
    local_delta,
    make_key,
    params_from_controls,
    save_keys,
)


def test_params_from_controls_mouth() -> None:
    ctrl = SemanticControls(mouth_open=0.8, brow_l=-0.04)
    p = params_from_controls(ctrl)
    assert abs(float(p[0]) - 0.8) < 1e-6
    assert abs(float(p[4]) + 0.04) < 1e-6


def test_key_weight_peaks_at_same_expression() -> None:
    a = params_from_controls(SemanticControls(mouth_open=0.7))
    b = params_from_controls(SemanticControls(mouth_open=0.72))
    c = params_from_controls(SemanticControls(mouth_open=0.1))
    assert key_weight(a, b) > 0.8
    assert key_weight(a, c) < 0.3


def test_make_key_stores_only_moved_slots() -> None:
    rest = neutral_keypoints()
    edited = rest.copy()
    edited[25, 1] += 0.08
    key = make_key(
        auto_kps=rest,
        edited_kps=edited,
        params=params_from_controls(SemanticControls(mouth_open=0.9)),
        roll_deg=0.0,
        name="Open",
    )
    assert 25 in key["slots"]
    assert 15 not in key["slots"]


def test_apply_keys_fades_in_near_matching_mouth() -> None:
    rest = neutral_keypoints()
    auto = rest.copy()
    edited = rest.copy()
    edited[25, 1] += 0.10
    key = make_key(
        auto_kps=auto,
        edited_kps=edited,
        params=params_from_controls(SemanticControls(mouth_open=0.85)),
        roll_deg=0.0,
    )
    near = apply_keys(
        rest.copy(),
        params_from_controls(SemanticControls(mouth_open=0.85)),
        [key],
    )
    far = apply_keys(
        rest.copy(),
        params_from_controls(SemanticControls(mouth_open=0.0)),
        [key],
    )
    assert abs(float(near[25, 1] - rest[25, 1]) - 0.10) < 0.02
    assert abs(float(far[25, 1] - rest[25, 1])) < 0.03


def test_local_delta_rotates_with_head_roll() -> None:
    auto = neutral_keypoints()
    edited = auto.copy()
    edited[21, 0] += 0.05
    delta = local_delta(auto, edited, roll_deg=90.0)
    # 90° local: +x world becomes -y local (image y down, standard rot).
    assert abs(float(delta[21, 1])) > abs(float(delta[21, 0]))


def test_keys_roundtrip(tmp_path) -> None:
    rest = neutral_keypoints()
    edited = rest.copy()
    edited[21, 1] -= 0.04
    key = make_key(
        auto_kps=rest,
        edited_kps=edited,
        params=params_from_controls(SemanticControls(mouth_open=0.5)),
        roll_deg=12.0,
        name="Lip",
    )
    ref = tmp_path / "hero.vtm"
    _write_pack(ref)
    assert save_keys(ref, [key]) == ref
    assert not (tmp_path / "hero.keys.json").exists()
    loaded = load_keys(ref)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "Lip"
    assert 21 in loaded[0]["slots"]
    from backend.character_pack import read_character_pack

    assert read_character_pack(ref).pose_keys[0]["name"] == "Lip"

    still = tmp_path / "hero.png"
    save_keys(still, [key])
    assert (tmp_path / "hero_pose_keys.json").is_file()
    assert load_keys(still)[0]["name"] == "Lip"


def _write_pack(dest) -> None:
    import numpy as np

    from backend.character_pack import write_character_pack

    write_character_pack(
        dest,
        name="Hero",
        preview_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        keypoints=neutral_keypoints(),
        ref_latent=np.zeros((4, 8, 8), dtype=np.float16),
        ref_face_latent=None,
        image_size=64,
        skip_crop=True,
    )


def test_legacy_key_sidecar_folds_into_pack_once(tmp_path) -> None:
    import json

    from backend.character_pack import read_character_pack, write_pack_pose_keys

    ref = tmp_path / "hero.vtm"
    _write_pack(ref)
    sidecar = tmp_path / "hero.keys.json"
    sidecar.write_text(json.dumps({"keys": [{"id": "a", "delta": [[0.1, 0.0]]}]}), encoding="utf-8")
    assert [k["id"] for k in load_keys(ref)] == ["a"]
    assert not sidecar.exists()
    assert read_character_pack(ref).version == 2
    assert [k["id"] for k in read_character_pack(ref).pose_keys] == ["a"]

    # Keys already in the pack win over a stale sidecar (which is still removed).
    write_pack_pose_keys(ref, [{"id": "pack", "delta": [[0.0, 0.2]]}])
    sidecar.write_text(json.dumps({"keys": [{"id": "old", "delta": [[1, 1]]}]}), encoding="utf-8")
    assert [k["id"] for k in load_keys(ref)] == ["pack"]
    assert not sidecar.exists()


def test_rest_bake_still_isolates_pointer_delta() -> None:
    rest = neutral_keypoints()
    base = rest.copy()
    base[25, 1] += 0.10
    edited = base.copy()
    edited[25, 0] += 0.04
    out = apply_overlay_drag_to_rest(rest, base, edited, [25])
    assert abs(float(out[25, 0]) - float(rest[25, 0]) - 0.04) < 1e-6


def _bare_runtime(tmp_path, rest: np.ndarray):
    import threading
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    adopted: list[np.ndarray] = []
    rt._lock = threading.Lock()
    rt._boot_lock = threading.Lock()
    rt._boot = {"ready": False, "running": False, "error": "", "stages": {}}
    rt._listeners = []
    rt._status = {
        "show_mesh": True,
        "pose_frozen": False,
        "pose_key_count": 0,
        "travel_box": {"enabled": True, "left": 0.16, "right": 0.16, "up": 0.3, "down": 0.24},
    }
    rt.engine = SimpleNamespace(
        _ref_keypoints=rest.copy(),
        compile_status="off",
        compile_detail="",
        keypoint_layout="hrnet_native",
        adopt_ref_keypoints=lambda kps, persist=True: adopted.append(np.asarray(kps).copy()) or kps,
    )
    rt._last_overlay_kps = rest.copy()
    rt._driven_keypoints = rest.copy()
    rt._last_image = None
    rt._mesh_edited = False
    rt._pose_frozen = False
    rt._pose_keys = []
    rt._freeze_auto_kps = None
    rt._freeze_params = None
    rt._freeze_roll = 0.0
    rt._prev_controls = SemanticControls(mouth_open=0.8)
    rt._ref_path = tmp_path / "hero.vtm"
    rt._ref_path.write_bytes(b"x")
    rt._drag_kp_idx = None
    rt._drag_slots = set()
    rt._drag_base_kps = None
    rt._drag_xy = None
    rt._last_mesh_emit_t = 0.0
    rt._tracking = False
    rt._lab_drive = False
    rt._adopted = adopted
    return rt


def test_frozen_mesh_release_does_not_bake_rest(tmp_path) -> None:
    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt.freeze_pose()
    edited = rest.copy()
    edited[25, 1] += 0.07
    rt._last_overlay_kps = edited
    rt._drag_slots = {25}
    rt._drag_base_kps = rest.copy()
    rt.mesh_release()
    assert rt._pose_frozen is True
    assert rt._mesh_edited is True
    np.testing.assert_allclose(rt.engine._ref_keypoints[25, :2], rest[25, :2], atol=1e-6)
    assert rt._adopted == []
