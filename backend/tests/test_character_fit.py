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

    # Track Lab re-detects: a different face, no nudges. Its rest reaches the
    # desk as the rest (a track ack), not only as whatever the overlay shows.
    rt._last_overlay_kps = neutral_keypoints().copy()
    rt._rest_kps = neutral_keypoints().copy()
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


# --- The rest a fit edit saves is the character's rest, never the overlay ---

_NUDGE = {"id": 18, "dx": 3.0, "dy": -2.0}


def _rest_runtime(tmp_path: Path, monkeypatch):
    """_edit_runtime with every character folder in tmp_path and a known rest."""
    monkeypatch.setattr("backend.paths.characters_dir", lambda: tmp_path)
    rt = _edit_runtime(tmp_path, monkeypatch)
    rest = neutral_keypoints().copy()
    rest[:, 3] = 1.0
    rt._rest_kps = rest.copy()
    return rt, rest


def _nudged(rest: np.ndarray) -> np.ndarray:
    """``rest`` as Track Lab draws it: the nudge on slot 18 (100 px canvas)."""
    shown = rest.copy()
    shown[18, 0] += _NUDGE["dx"] * 2.0 / 100.0
    shown[18, 1] += _NUDGE["dy"] * 2.0 / 100.0
    return shown


def test_a_point_move_while_tracking_saves_the_rest_not_the_live_pose(tmp_path: Path, monkeypatch) -> None:
    """The 50 ms lab poll and the stream write the live / generated pose into
    the overlay; a fit edit saved that overlay into the .vtm as the rest."""
    from backend.pose_controller import pixels_to_normalized
    from backend.stream import StreamRuntime

    rt, rest = _rest_runtime(tmp_path, monkeypatch)
    live = rest.copy()
    live[:28, 0] += 0.2  # head turned
    rt._last_overlay_kps = live.copy()
    # Track Lab is tracking: its reply carries the live frame.
    rt._lab_packet_from_ack = lambda ack: {"live": True, "keypoints": [], "point_offsets": []}
    StreamRuntime.move_character_point(rt, 12, 30.0, 40.0)
    want = rest.copy()
    want[12, :2] = pixels_to_normalized(30.0, 40.0, 100, 100)
    saved = read_character_pack(tmp_path / "hero.vtm").keypoints
    np.testing.assert_allclose(saved[:, :2], want[:, :2], atol=1e-6)
    np.testing.assert_allclose(rt._rest_kps[:, :2], want[:, :2], atol=1e-6)


def test_nudges_are_never_baked_into_the_saved_rest(tmp_path: Path, monkeypatch) -> None:
    """Track Lab's frame has the nudges on. Saved as the rest, the next load
    sent it with the nudges beside it and they applied twice."""
    from backend.stream import StreamRuntime

    rt, rest = _rest_runtime(tmp_path, monkeypatch)
    rt._last_overlay_kps = _nudged(rest)
    rt.lab_offsets = [dict(_NUDGE)]
    # Track Lab's own rest frame (not live), nudge on.
    rt._lab_packet_from_ack = lambda ack: {
        "keypoints": [],
        "image_wh": [100, 100],
        "point_offsets": list(rt.lab_offsets),
    }
    for _ in range(3):
        StreamRuntime.move_character_point(rt, 12, 30.0, 40.0)
    saved = read_character_pack(tmp_path / "hero.vtm").keypoints
    np.testing.assert_allclose(saved[18, :2], rest[18, :2], atol=1e-6)
    # The overlay and the model still show the still as Track Lab draws it.
    np.testing.assert_allclose(rt._last_overlay_kps[18, :2], _nudged(rest)[18, :2], atol=1e-6)
    # Next load: the rest goes to Track Lab once, the nudge beside it once.
    rt._still_image = rt._last_image
    body = StreamRuntime._packaged_lab_rest(rt, {"commands": ["set_rest"]})
    assert body is not None
    x18 = (float(rest[18, 0]) + 1.0) * 0.5 * 100
    assert abs(body["points"][18][0] - x18) < 1e-3
    assert body["point_offsets"] == [_NUDGE]


def test_the_rest_read_from_track_lab_has_its_edits_but_not_its_nudges(tmp_path: Path, monkeypatch) -> None:
    """Track Lab's reply frame: its own edit on a joint (kept), the nudge on
    (taken off), a mouth shape being authored (the status rest wins) and
    0.01 px rounding everywhere (the held rest wins)."""
    from backend.pose_controller import pixels_to_normalized
    from backend.stream import StreamRuntime

    rt, rest = _rest_runtime(tmp_path, monkeypatch)
    lab = rest.copy()
    lab[33, :2] = (0.3, 0.6)  # moved in Track Lab
    frame = _nudged(lab)
    frame[22, 1] += 0.1  # the mouth shape on screen in Track Lab
    frame[:, :2] += 0.004 * 2.0 / 100.0  # Track Lab's rounding
    face = [[(float(lab[i, 0]) + 1.0) * 50.0, (float(lab[i, 1]) + 1.0) * 50.0, 1.0] for i in range(28)]
    del rt.adopt_lab_overlay
    rt._lab_overlay_keypoints = lambda packet=None: frame.copy()
    rt.lab_offsets = [dict(_NUDGE)]
    rt._lab_packet_from_ack = lambda ack: {
        "keypoints": [],
        "image_wh": [100, 100],
        "rest": [list(row) for row in face],
        "point_offsets": list(rt.lab_offsets),
    }
    StreamRuntime.move_character_point(rt, 12, 30.0, 40.0)
    want = rest.copy()
    want[33, :2] = (0.3, 0.6)
    want[12, :2] = pixels_to_normalized(30.0, 40.0, 100, 100)
    saved = read_character_pack(tmp_path / "hero.vtm").keypoints
    np.testing.assert_allclose(saved[:, :2], want[:, :2], atol=1e-4)
    for slot in (0, 18, 22, 28, 31):
        assert np.array_equal(saved[slot], rest[slot]), slot


def test_skeleton_move_saves_rest_joints_without_nudges(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    rt, rest = _rest_runtime(tmp_path, monkeypatch)
    nudge = {"id": 34, "dx": 4.0, "dy": 0.0}
    shown = rest.copy()
    shown[34, 0] += 4.0 * 2.0 / 100.0
    rt._last_overlay_kps = shown.copy()
    rt._lab_packet_from_ack = lambda ack: {"keypoints": [], "image_wh": [100, 100], "point_offsets": [nudge]}
    StreamRuntime.move_character_skeleton(rt, 32, 20.0, 70.0)
    joints = {row["id"]: row for row in read_character_fit("hero")["skeleton"]}
    assert joints[34]["x"] == round(float(rest[34, 0]), 5)
    saved = read_character_pack(tmp_path / "hero.vtm").keypoints
    np.testing.assert_allclose(saved[34, :2], rest[34, :2], atol=1e-6)


def test_track_lab_rounding_is_not_read_as_a_moved_point(tmp_path: Path, monkeypatch) -> None:
    """The restore compared points at 1e-5 (0.004 px); Track Lab sends 0.01 px,
    so a load re-sent the user's points and rewrote the pack every time."""
    from backend.stream import StreamRuntime

    rt, rest = _rest_runtime(tmp_path, monkeypatch)
    rt._lab_image_wh = (768, 768)
    update_character_fit("hero", {"points": {"12": [round(float(rest[12, 0]), 5), round(float(rest[12, 1]), 5)]}})
    echoed = rest.copy()
    echoed[12, 0] += 0.01 * 2.0 / 768.0  # Track Lab's rounding
    rt._rest_kps = echoed.copy()
    rt._last_overlay_kps = echoed.copy()
    pack = tmp_path / "hero.vtm"
    stamp = pack.stat().st_mtime_ns
    before = read_character_pack(pack).keypoints.copy()
    rt.sent.clear()
    assert StreamRuntime._apply_character_fit(rt) is False
    assert [op for op, _ in rt.sent] == []
    np.testing.assert_array_equal(read_character_pack(pack).keypoints, before)
    assert pack.stat().st_mtime_ns == stamp


def test_a_load_never_takes_the_live_frame_as_the_rest(tmp_path: Path, monkeypatch) -> None:
    """A set_rest reply without keypoints made the overlay fall back to
    GET /frame, the live pose while tracking, and the model took it as the
    character's reference."""
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    rt, rest = _rest_runtime(tmp_path, monkeypatch)
    del rt._lab_packet_from_ack
    del rt.adopt_lab_overlay
    rt._still_image = rt._last_image
    rt._lab_seen_generation = 0
    rt._lab_overlay_gen = 0
    adopted: list[np.ndarray] = []
    rt.engine = SimpleNamespace(
        _ref_keypoints=rest.copy(),
        adopt_ref_keypoints=lambda kps, **kw: adopted.append(np.asarray(kps).copy()),
    )
    live = rest.copy()
    live[:28, 1] += 0.15

    def get_frame(frame=None):
        rt._last_overlay_kps = live.copy()
        return live.copy()

    rt._lab_overlay_keypoints = get_frame

    class _Lab:
        def status(self, merge_frame=False):
            return {"online": True, "ready": False, "commands": ["set_rest", "track"]}

        def put_source(self, path):
            return {"ok": True, "status": {"generation": 2}}

    monkeypatch.setattr("backend.lab_harness.lab", _Lab())
    rt._same_lab_still = lambda: False
    rt._write_lab_source = lambda: "track_lab/input/source.png"
    rt._adopt_lab_hair = lambda packet=None: True
    rt._lab_ack = lambda op, body=None: rt.sent.append((op, body)) or {
        "ok": True,
        "status": {"generation": 2, "point_offsets": []},
    }
    StreamRuntime._sync_lab_character(rt)
    assert rt.sent[0][0] == "set_rest"
    np.testing.assert_allclose(rt._rest_kps[:, :2], rest[:, :2], atol=1e-6)
    assert adopted
    np.testing.assert_allclose(adopted[-1][:, :2], rest[:, :2], atol=1e-6)


def test_desk_mesh_drag_saves_the_rest_without_nudges(tmp_path: Path) -> None:
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints().copy()
    rt = _bare_runtime(tmp_path, _nudged(rest))
    persisted: list[tuple[bool, np.ndarray]] = []

    def adopt(kps, persist=True, **kw):
        persisted.append((persist, np.asarray(kps).copy()))
        rt.engine._ref_keypoints = np.asarray(kps).copy()

    rt.engine.adopt_ref_keypoints = adopt
    rt._rest_kps = rest.copy()
    base = _nudged(rest)
    edited = base.copy()
    edited[21, 0] += 0.05
    rt._last_overlay_kps = edited
    rt._drag_slots = {21}
    rt._drag_base_kps = base.copy()
    rt.mesh_release()
    saved = [kps for persist, kps in persisted if persist]
    assert len(saved) == 1
    np.testing.assert_allclose(saved[0][18, :2], rest[18, :2], atol=1e-6)
    np.testing.assert_allclose(saved[0][21, 0], rest[21, 0] + 0.05, atol=1e-6)
    # The model keeps drawing the still with the nudge on.
    np.testing.assert_allclose(rt.engine._ref_keypoints[18, :2], _nudged(rest)[18, :2], atol=1e-6)
    np.testing.assert_allclose(rt._rest_kps, saved[0], atol=1e-6)
