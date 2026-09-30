from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.character_fit import (
    POINT_SLOTS,
    SKELETON_LABELS,
    build_fit_view,
    read_character_fit,
    replace_pack_keypoints,
    update_character_fit,
)
from backend.character_pack import read_character_pack, write_character_pack
from backend.engine import neutral_keypoints
from backend.travel_box import default_travel_box


def _tiny_pack_payload() -> dict:
    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (40, 80, 60)
    return {
        "name": "Fit",
        "preview_rgb": preview,
        "keypoints": neutral_keypoints(),
        "ref_latent": np.zeros((1, 4, 8, 8), dtype=np.float16),
        "ref_face_latent": np.zeros((1, 4, 4, 4), dtype=np.float16),
        "image_size": 768,
        "skip_crop": True,
        "source_name": "fit.png",
    }


def test_build_fit_view_puts_skeleton_and_body_in_pixels() -> None:
    kps = neutral_keypoints()
    width, height = 200, 160
    view = build_fit_view(
        width=width,
        height=height,
        keypoints=kps,
        hair_norm=[],
        box=default_travel_box(),
    )
    assert view["width"] == width
    assert view["height"] == height
    assert len(view["skeleton"]) == len(SKELETON_LABELS)
    for joint in view["skeleton"]:
        assert 0.0 <= joint["x"] <= width
        assert 0.0 <= joint["y"] <= height
        assert joint["id"] in SKELETON_LABELS
    body = view["boxes"]["body_tight"]
    assert body is not None
    assert len(body) == 4
    assert body[0] < body[2]
    assert body[1] < body[3]
    assert 0.0 <= body[0] <= width
    assert 0.0 <= body[3] <= height


def test_build_fit_view_lists_face_iris_and_body_points() -> None:
    kps = neutral_keypoints().copy()
    kps[30, 3] = 0.0
    view = build_fit_view(width=200, height=160, keypoints=kps, hair_norm=[], box=default_travel_box())
    by_id = {row["id"]: row for row in view["points"]}
    assert set(by_id) == set(POINT_SLOTS)
    assert by_id[13]["group"] == "face" and by_id[13]["label"] == "EYE.L.OUT"
    assert by_id[28]["group"] == "iris"
    assert by_id[31]["group"] == "body" and by_id[31]["label"] == SKELETON_LABELS[31]
    hidden = kps.copy()
    hidden[13, 3] = 0.0
    view = build_fit_view(width=200, height=160, keypoints=hidden, hair_norm=[], box=default_travel_box())
    assert 13 not in {row["id"] for row in view["points"]}


def test_update_character_fit_roundtrip(tmp_path: Path) -> None:
    update_character_fit("hero", {"hair": [{"class": "hair_middle", "polygon": [[0, 0], [1, 0], [0, 1]]}]}, dest_dir=tmp_path)
    update_character_fit("hero", {"skeleton": [{"id": 31, "x": 0.1, "y": 0.2}]}, dest_dir=tmp_path)
    saved = read_character_fit("hero", dest_dir=tmp_path)
    assert saved["hair"][0]["class"] == "hair_middle"
    assert saved["skeleton"][0]["id"] == 31
    assert saved["skeleton"][0]["x"] == 0.1


def test_replace_pack_keypoints_swaps_npy(tmp_path: Path) -> None:
    dest = tmp_path / "fit.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    edited = neutral_keypoints().copy()
    edited[31, 0] = 0.42
    edited[31, 1] = -0.17
    replace_pack_keypoints(dest, edited)
    pack = read_character_pack(dest)
    np.testing.assert_allclose(pack.keypoints[31, :2], (0.42, -0.17), atol=1e-5)
    assert pack.keypoints.shape == (37, 4)


def test_fit_lives_inside_the_pack_and_absorbs_the_sidecar(tmp_path: Path) -> None:
    from backend.character_fit import character_fit_path, write_character_fit
    from backend.character_pack import read_pack_fit

    dest = tmp_path / "hero.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    legacy = character_fit_path("hero", dest_dir=tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"hair": [{"class": "hair_left", "polygon": [[0, 0], [1, 0], [0, 1]]}]}', encoding="utf-8")

    assert read_character_fit("hero", dest_dir=tmp_path)["hair"][0]["class"] == "hair_left"
    box = {**default_travel_box(), "left": 0.3}
    update_character_fit("hero", {"travel_box": box}, dest_dir=tmp_path)

    assert not legacy.exists()
    inside = read_pack_fit(dest)
    assert inside["hair"][0]["class"] == "hair_left"
    assert inside["travel_box"]["left"] == 0.3
    write_character_fit("hero", inside, dest_dir=tmp_path)
    assert read_character_pack(dest).fit["travel_box"]["left"] == 0.3


def test_pack_fit_survives_keypoint_swap(tmp_path: Path) -> None:
    dest = tmp_path / "fit.vtm"
    box = {**default_travel_box(), "up": 0.9}
    write_character_pack(dest, **_tiny_pack_payload(), fit={"travel_box": box})
    replace_pack_keypoints(dest, neutral_keypoints())
    assert read_character_pack(dest).fit["travel_box"]["up"] == 0.9


def _limiter_runtime(ident: str):
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    import threading

    rt._lock = threading.Lock()
    rt._status = {"character_id": ident, "travel_box": default_travel_box()}
    rt._emitted = []
    rt._emit = rt._emitted.append
    rt._lab_seen_online = False
    rt._travel_from_desk = False
    return rt


def test_each_character_loads_its_own_limiters(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    monkeypatch.setattr("backend.character_pack.characters_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.ui_session.save_ui_session", lambda **kwargs: None)
    write_character_pack(
        tmp_path / "a.vtm", **_tiny_pack_payload(), fit={"travel_box": {**default_travel_box(), "left": 0.2}}
    )
    write_character_pack(
        tmp_path / "b.vtm", **_tiny_pack_payload(), fit={"travel_box": {**default_travel_box(), "left": 0.9}}
    )

    rt = _limiter_runtime("a")
    StreamRuntime._apply_character_limiters(rt)
    assert rt._status["travel_box"]["left"] == 0.2
    assert rt._travel_from_desk is True

    rt._status["character_id"] = "b"
    StreamRuntime._apply_character_limiters(rt)
    assert rt._status["travel_box"]["left"] == 0.9

    StreamRuntime._save_character_limiters(rt, {**rt._status["travel_box"], "left": 0.5})
    assert read_character_pack(tmp_path / "b.vtm").fit["travel_box"]["left"] == 0.5
    assert read_character_pack(tmp_path / "a.vtm").fit["travel_box"]["left"] == 0.2


def _unfitted_runtime(tmp_path: Path, monkeypatch, ident: str = "new"):
    from backend.stream import StreamRuntime

    monkeypatch.setattr("backend.character_pack.characters_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.ui_session.save_ui_session", lambda **kwargs: None)
    write_character_pack(tmp_path / f"{ident}.vtm", **_tiny_pack_payload())
    rt = _limiter_runtime(ident)
    # The last character's limits are still on the desk.
    rt._status["travel_box"] = {**default_travel_box(), "down": 0.4}
    rt.status = lambda: dict(rt._status)
    rt.lab_calls = []
    StreamRuntime._apply_character_limiters(rt)
    return rt


def _lab_fits(rt, box: dict) -> None:
    def ack(op: str, body: dict | None = None) -> dict:
        rt.lab_calls.append((op, body))
        return {"ok": True, "status": {"travel_box": box}}

    rt._lab_ack = ack


def test_new_character_gets_limiters_fitted_not_the_last_ones(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _unfitted_runtime(tmp_path, monkeypatch)
    assert "travel_box" not in read_character_pack(tmp_path / "new.vtm").fit
    fitted = {**default_travel_box(), "left": 0.33, "turn_left": 18.0, "turn_right": 18.0}
    _lab_fits(rt, fitted)
    StreamRuntime._fit_new_character_limiters(rt)
    assert rt.lab_calls == [("fit_travel", {"from": "default"})]
    assert rt._status["travel_box"]["left"] == 0.33
    assert read_character_pack(tmp_path / "new.vtm").fit["travel_box"]["left"] == 0.33
    # Saved now: the next load installs them and does not fit again.
    rt.lab_calls.clear()
    StreamRuntime._apply_character_limiters(rt)
    StreamRuntime._fit_new_character_limiters(rt)
    assert rt.lab_calls == []
    assert rt._status["travel_box"]["turn_left"] == 18.0


def test_limiter_fit_waits_while_track_lab_is_offline(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _unfitted_runtime(tmp_path, monkeypatch)

    def offline(op: str, body: dict | None = None) -> dict:
        raise RuntimeError("Track Lab is not running")

    rt._lab_ack = offline
    StreamRuntime._fit_new_character_limiters(rt)
    assert rt._limiters_unfitted == "new"
    assert "travel_box" not in read_character_pack(tmp_path / "new.vtm").fit
    # Track Lab comes up (boot sync): the fit runs then.
    _lab_fits(rt, {**default_travel_box(), "up": 0.7})
    StreamRuntime._fit_new_character_limiters(rt)
    assert read_character_pack(tmp_path / "new.vtm").fit["travel_box"]["up"] == 0.7


def test_limiter_edit_before_the_fit_is_kept(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _unfitted_runtime(tmp_path, monkeypatch)
    StreamRuntime._save_character_limiters(rt, {**default_travel_box(), "left": 0.9})
    _lab_fits(rt, {**default_travel_box(), "left": 0.2})
    StreamRuntime._fit_new_character_limiters(rt)
    assert rt.lab_calls == []
    assert read_character_pack(tmp_path / "new.vtm").fit["travel_box"]["left"] == 0.9


def test_fit_button_keeps_look_and_saves_even_an_unchanged_box(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime
    from backend.travel_box import normalize_travel_box

    rt = _unfitted_runtime(tmp_path, monkeypatch)
    same = normalize_travel_box(rt._status["travel_box"])
    _lab_fits(rt, same)
    StreamRuntime.fit_character_limiters(rt)
    # No "from": Track Lab keeps the current look / eyes / size.
    assert rt.lab_calls == [("fit_travel", {})]
    assert read_character_pack(tmp_path / "new.vtm").fit["travel_box"] == same
    assert rt._travel_from_desk is True


def _overlay_runtime():
    import threading

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._pose_frozen = False
    rt._mesh_edited = False
    rt._last_image = None
    rt._emit = lambda event: None
    return rt


def test_status_poll_from_before_a_fit_edit_cannot_undo_it() -> None:
    from backend.stream import StreamRuntime, fit_edit

    rt = _overlay_runtime()
    adopted: list = []
    rt._lab_overlay_keypoints = lambda frame=None: adopted.append(frame) or neutral_keypoints()
    before = StreamRuntime.fit_generation(rt)
    during: list[bool] = []

    @fit_edit
    def edit(self) -> None:
        during.append(StreamRuntime.fit_poll_stale(self, StreamRuntime.fit_generation(self)))

    edit(rt)
    assert during == [True]
    assert StreamRuntime.adopt_lab_overlay(rt, {"keypoints": []}, fit_gen=before) is False
    assert adopted == []
    after = StreamRuntime.fit_generation(rt)
    assert StreamRuntime.adopt_lab_overlay(rt, {"keypoints": []}, fit_gen=after) is True
    # The fit endpoints and the track thread pass no token and are never held back.
    assert StreamRuntime.adopt_lab_overlay(rt, {"keypoints": []}) is True
    assert len(adopted) == 2


def test_point_move_saves_track_labs_whole_rest(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from PIL import Image

    from backend.pose_controller import pixels_to_normalized
    from backend.stream import StreamRuntime

    dest = tmp_path / "hero.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    rt = _overlay_runtime()
    rt._last_image = Image.new("RGB", (100, 100))
    rt._frame_payload = lambda image, kps: {}
    rt.character_fit = lambda: {}
    rt._ref_path = dest
    rt.engine = SimpleNamespace(adopt_ref_keypoints=lambda kps, **kw: None, _ref_keypoints_session_base=None)
    # Track Lab holds an earlier edit (slot 12) that a stale poll dropped from the desk.
    lab_rest = neutral_keypoints().copy()
    lab_rest[12, :2] = (0.1, 0.2)
    rt._last_overlay_kps = neutral_keypoints().copy()

    def lab_overlay(frame=None):
        rt._last_overlay_kps = lab_rest.copy()
        return lab_rest.copy()

    rt._lab_overlay_keypoints = lab_overlay
    rt._desk_px_to_lab = lambda x, y: (x, y)
    sent: list = []
    rt._lab_ack = lambda op, body=None: sent.append((op, body)) or {"ok": True}
    rt._lab_packet_from_ack = lambda ack: {"keypoints": []}

    StreamRuntime.move_character_point(rt, 18, 60.0, 40.0)

    assert sent == [("set_rest_point", {"id": 18, "x": 60.0, "y": 40.0})]
    saved = read_character_pack(dest).keypoints
    np.testing.assert_allclose(saved[12, :2], (0.1, 0.2), atol=1e-5)
    np.testing.assert_allclose(saved[18, :2], pixels_to_normalized(60.0, 40.0, 100, 100), atol=1e-5)
    # A later mesh reset returns to the fitted rest, not the pre-fit one.
    np.testing.assert_allclose(rt.engine._ref_keypoints_session_base, saved, atol=1e-5)


def test_sidecar_folds_into_pack_once_and_is_deleted(tmp_path: Path) -> None:
    import json
    import zipfile

    from backend.character_fit import character_fit_path
    from backend.character_pack import FIT_NAME, peek_character_manifest

    dest = tmp_path / "hero.vtm"
    write_character_pack(dest, **_tiny_pack_payload(), fit={"travel_box": {**default_travel_box(), "left": 0.2}})
    legacy = character_fit_path("hero", dest_dir=tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        json.dumps({"travel_box": {"left": 0.8}, "skeleton": [{"id": 31, "x": 1.0, "y": 2.0}]}),
        encoding="utf-8",
    )
    fit = read_character_fit("hero", dest_dir=tmp_path)
    assert not legacy.exists()
    assert fit["travel_box"]["left"] == 0.2  # the pack is the source of truth
    assert fit["skeleton"][0]["id"] == 31
    with zipfile.ZipFile(dest) as zf:
        inside = json.loads(zf.read(FIT_NAME))
    assert inside == fit
    assert peek_character_manifest(dest)["version"] == 2
    stamp = dest.stat().st_mtime_ns
    assert read_character_fit("hero", dest_dir=tmp_path) == fit
    assert dest.stat().st_mtime_ns == stamp  # later reads do not rewrite


def test_sidecar_kept_when_pack_is_broken(tmp_path: Path) -> None:
    from backend.character_fit import character_fit_path

    (tmp_path / "hero.vtm").write_bytes(b"not a zip")
    legacy = character_fit_path("hero", dest_dir=tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"hair": []}', encoding="utf-8")
    assert read_character_fit("hero", dest_dir=tmp_path) == {"hair": []}
    assert legacy.exists()


def _edit_runtime(tmp_path: Path, monkeypatch):
    """A loaded character whose Track Lab replies to every command."""
    from types import SimpleNamespace

    from PIL import Image

    monkeypatch.setattr("backend.character_pack.characters_dir", lambda: tmp_path)
    write_character_pack(tmp_path / "hero.vtm", **_tiny_pack_payload())
    rt = _overlay_runtime()
    rt._status = {"character_id": "hero"}
    rt._last_image = Image.new("RGB", (100, 100))
    rt._frame_payload = lambda image, kps: {}
    rt.character_fit = lambda: {}
    rt._ref_path = tmp_path / "hero.vtm"
    rt.engine = SimpleNamespace(adopt_ref_keypoints=lambda kps, **kw: None, _ref_keypoints_session_base=None)
    rt._last_overlay_kps = neutral_keypoints().copy()
    rt._lab_image_wh = (100, 100)
    rt._desk_px_to_lab = lambda x, y: (x, y)
    rt._lab_packet_from_ack = lambda ack: {"keypoints": []}
    rt.adopt_lab_overlay = lambda packet, **kw: True
    rt.lab_offsets = []
    rt.sent = []

    def ack(op: str, body: dict | None = None) -> dict:
        rt.sent.append((op, body))
        if op == "set_point":
            rt.lab_offsets = [{"id": int(body["id"]), "dx": 3.0, "dy": -2.0}]
        if op in ("reset_points", "set_rest_point"):
            rt.lab_offsets = [r for r in rt.lab_offsets if op == "set_rest_point" and r["id"] != body["id"]]
        return {"ok": True, "status": {"point_offsets": list(rt.lab_offsets)}}

    rt._lab_ack = ack
    return rt


def test_a_redetect_never_overrides_the_users_points_or_nudges(tmp_path: Path, monkeypatch) -> None:
    """Track Lab re-detecting the still put its own face back and the restore
    saved that into the pack; a moved eye point and a nudge were lost."""
    from backend.pose_controller import pixels_to_normalized
    from backend.stream import StreamRuntime

    rt = _edit_runtime(tmp_path, monkeypatch)
    StreamRuntime.move_character_point(rt, 12, 30.0, 40.0)
    StreamRuntime._nudge_lab_point(rt, 18, 70.0, 45.0)
    fit = read_character_fit("hero")
    assert "12" in fit["points"]
    assert fit["point_offsets"] == [{"id": 18, "dx": 3.0, "dy": -2.0}]

    # Track Lab re-detects: a different face, no nudges.
    rt._last_overlay_kps = neutral_keypoints().copy()
    rt.lab_offsets = []
    rt.sent.clear()
    assert StreamRuntime._apply_character_fit(rt) is True
    want = pixels_to_normalized(30.0, 40.0, 100, 100)
    np.testing.assert_allclose(rt._last_overlay_kps[12, :2], want, atol=1e-5)
    np.testing.assert_allclose(read_character_pack(tmp_path / "hero.vtm").keypoints[12, :2], want, atol=1e-5)
    ops = [op for op, _ in rt.sent]
    assert ops.index("set_rest_point") < ops.index("set_offsets")  # the move would drop the nudge
    assert ("set_offsets", {"point_offsets": [{"id": 18, "dx": 3.0, "dy": -2.0}]}) in rt.sent

    # Already in place: nothing is re-sent for the point.
    rt.sent.clear()
    StreamRuntime._apply_character_fit(rt)
    assert [op for op, _ in rt.sent] == ["set_offsets"]


def test_a_packaged_rest_brings_its_nudges(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _edit_runtime(tmp_path, monkeypatch)
    StreamRuntime._nudge_lab_point(rt, 18, 70.0, 45.0)
    rt._still_image = rt._last_image
    rt.engine._ref_keypoints = neutral_keypoints().copy()
    body = StreamRuntime._packaged_lab_rest(rt, {"commands": ["set_rest"]})
    assert body is not None and body["point_offsets"] == [{"id": 18, "dx": 3.0, "dy": -2.0}]


def test_a_reset_is_the_users_choice_and_is_kept(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt = _edit_runtime(tmp_path, monkeypatch)
    StreamRuntime._nudge_lab_point(rt, 18, 70.0, 45.0)
    rt._freeze_auto_kps = None
    rt._drag_kp_idx = None
    rt._drag_slots = set()
    rt._drag_base_kps = None
    rt._drag_xy = None
    rt._lab_drive = True
    rt._tracking = False
    StreamRuntime.mesh_reset(rt)
    assert read_character_fit("hero")["point_offsets"] == []


def test_detected_hair_never_replaces_painted_hair(tmp_path: Path, monkeypatch) -> None:
    from backend.character_fit import update_character_fit
    from backend.stream import StreamRuntime

    rt = _edit_runtime(tmp_path, monkeypatch)
    painted = [{"class": "hair_middle", "polygon": [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]}]
    update_character_fit("hero", {"hair": painted})
    rt._last_lab_hair = [{"class": "hair_left", "polygon": [[0.5, 0.5], [0.6, 0.5], [0.5, 0.6]]}]
    StreamRuntime._store_pack_hair(rt)
    assert read_character_fit("hero")["hair"] == painted
