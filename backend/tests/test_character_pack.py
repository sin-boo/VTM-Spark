from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.character_pack import (
    CharacterPackError,
    character_card,
    character_still_path,
    ensure_character_still,
    list_character_files,
    peek_character_manifest,
    read_character_pack,
    rename_character_pack,
    resolve_character_id,
    stage_create_still,
    unique_character_path,
    validate_character_pack,
    write_character_pack,
)
from backend.engine import StreamEngine, neutral_keypoints


def _tiny_pack_payload() -> dict:
    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (40, 80, 60)
    kps = neutral_keypoints()
    latent = np.zeros((1, 4, 8, 8), dtype=np.float16)
    face = np.zeros((1, 4, 4, 4), dtype=np.float16)
    return {
        "name": "Gigi",
        "preview_rgb": preview,
        "keypoints": kps,
        "ref_latent": latent,
        "ref_face_latent": face,
        "image_size": 768,
        "skip_crop": True,
        "source_name": "gigi.png",
    }


def test_character_pack_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "gigi.vtm"
    payload = _tiny_pack_payload()
    write_character_pack(dest, **payload)
    pack = read_character_pack(dest)
    assert pack.name == "Gigi"
    assert pack.image_size == 768
    assert pack.skip_crop is True
    assert pack.has_face_latent is True
    assert pack.keypoints.shape == (37, 4)
    np.testing.assert_allclose(pack.keypoints, payload["keypoints"], atol=1e-5)
    assert pack.ref_latent.shape == (1, 4, 8, 8)
    manifest = validate_character_pack(dest)
    assert manifest["format"] == "vtm-character"


def test_character_pack_rejects_garbage(tmp_path: Path) -> None:
    junk = tmp_path / "nope.vtm"
    junk.write_bytes(b"not a zip")
    with pytest.raises(CharacterPackError):
        read_character_pack(junk)


def test_unique_character_path_and_list(tmp_path: Path) -> None:
    payload = _tiny_pack_payload()
    first = unique_character_path("Gigi Mood", dest_dir=tmp_path)
    write_character_pack(first, **payload)
    second = unique_character_path("Gigi Mood", dest_dir=tmp_path)
    assert first.name == "Gigi-Mood.vtm"
    assert second.name == "Gigi-Mood-2.vtm"
    write_character_pack(second, **payload)
    files = list_character_files(dest_dir=tmp_path)
    assert [p.name for p in files] == ["Gigi-Mood-2.vtm", "Gigi-Mood.vtm"]
    found = resolve_character_id("Gigi-Mood", dest_dir=tmp_path)
    assert found == first
    card = character_card(first)
    assert card["id"] == "Gigi-Mood"
    assert "/Gigi-Mood/preview" in card["preview_url"]
    assert "v=" in card["preview_url"]
    still = character_still_path("Gigi-Mood", dest_dir=tmp_path)
    assert still.is_file()
    assert still.parent.name == "Gigi-Mood"
    assert resolve_character_id("Gigi-Mood", dest_dir=tmp_path) == first
    assert resolve_character_id("Gigi", dest_dir=tmp_path) in {first, second}
    assert ensure_character_still(first, dest_dir=tmp_path) == still


def test_add_character_copies_pack_into_library(tmp_path: Path) -> None:
    from backend.stream import StreamRuntime

    src = tmp_path / "import.vtm"
    write_character_pack(src, **_tiny_pack_payload())
    dest_dir = tmp_path / "library"
    dest_dir.mkdir()
    runtime = object.__new__(StreamRuntime)
    runtime.status = lambda: {}
    runtime._character_card_safe = lambda path: character_card(path)
    result = StreamRuntime.add_character(runtime, src, dest_dir=dest_dir)
    files = list_character_files(dest_dir=dest_dir)
    assert [p.name for p in files] == ["Gigi.vtm"]
    assert result["character"]["id"] == "Gigi"
    assert result["character"]["name"] == "Gigi"

    StreamRuntime.add_character(runtime, src, dest_dir=dest_dir)
    names = [p.name for p in list_character_files(dest_dir=dest_dir)]
    assert names == ["Gigi-2.vtm", "Gigi.vtm"]


def test_rename_character_pack_keeps_id(tmp_path: Path) -> None:
    dest = tmp_path / "Gigi.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    card = rename_character_pack(dest, "  Blue  Hair  ")
    assert dest.name == "Gigi.vtm"
    assert card["id"] == "Gigi"
    assert card["name"] == "Blue Hair"
    assert peek_character_manifest(dest)["name"] == "Blue Hair"
    with pytest.raises(CharacterPackError):
        rename_character_pack(dest, "   ")


def test_engine_load_encoded_reference_skips_vae(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _boom(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("encode_reference should not run")

    monkeypatch.setattr("inference_keypoint.encode_reference", _boom)
    engine = StreamEngine()
    engine._ready = True
    engine.model = None
    engine.device = __import__("torch").device("cpu")
    engine.keypoint_layout = "schema"
    dest = tmp_path / "packed.vtm"
    payload = _tiny_pack_payload()
    write_character_pack(dest, **payload)
    pack = read_character_pack(dest)
    engine.load_encoded_reference(
        keypoints=pack.keypoints,
        ref_latent=pack.ref_latent,
        ref_face_latent=pack.ref_face_latent,
        skip_crop=pack.skip_crop,
        path=dest,
    )
    assert calls == []
    assert engine._ref_latent is not None
    assert engine._ref_face_latent is not None
    assert engine._ref_path == dest
    assert engine._ref_pose_source == "character_pack"
    assert engine._ref_keypoints is not None
    assert engine._ref_keypoints_model.shape == (37, 4)


def test_discard_sidecar_keypoints(tmp_path: Path) -> None:
    from backend.engine import discard_sidecar_keypoints, sidecar_keypoints_npy_path

    still = tmp_path / "character_create_abc.png"
    still.write_bytes(b"png")
    side = sidecar_keypoints_npy_path(still)
    np.save(side, np.zeros((37, 4), dtype=np.float32))
    assert side.is_file()
    discard_sidecar_keypoints(still)
    assert not side.is_file()


def test_stage_create_still_is_unique(tmp_path: Path) -> None:
    a = stage_create_still(b"one", suffix=".png", dest_dir=tmp_path)
    b = stage_create_still(b"two", suffix=".png", dest_dir=tmp_path)
    assert a.name.startswith("character_create_")
    assert b.name.startswith("character_create_")
    assert a != b
    assert a.read_bytes() == b"one"
    assert b.read_bytes() == b"two"
    a = stage_create_still(b"one", suffix=".png", dest_dir=tmp_path)
    b = stage_create_still(b"two", suffix=".png", dest_dir=tmp_path)
    assert a.name.startswith("character_create_")
    assert b.name.startswith("character_create_")
    assert a != b
    assert a.read_bytes() == b"one"
    assert b.read_bytes() == b"two"


def test_ensure_character_still_refreshes_when_pack_newer(tmp_path: Path) -> None:
    dest = tmp_path / "gigi.vtm"
    payload = _tiny_pack_payload()
    write_character_pack(dest, **payload)
    still = ensure_character_still(dest, dest_dir=tmp_path)
    still.write_bytes(b"stale-preview-bytes")
    older = dest.stat().st_mtime - 20
    import os

    os.utime(still, (older, older))
    payload["preview_rgb"][:, :] = (9, 9, 9)
    write_character_pack(dest, **payload)
    refreshed = ensure_character_still(dest, dest_dir=tmp_path)
    assert refreshed == still
    data = still.read_bytes()
    assert data != b"stale-preview-bytes"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_reset_character_runtime_drops_old_still() -> None:
    import threading

    from PIL import Image

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._listeners = []
    rt._status = {
        "show_mesh": False,
        "show_hair": False,
        "travel_box": {"enabled": False},
    }
    rt.status = lambda: dict(rt._status)
    rt._body_lost = False
    rt._mouth_snapped = False
    rt._travel_silhouette = None
    rt._travel_silhouette_tight = None
    rt._travel_ref_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rt.engine = type("E", (), {"_ref_keypoints": None})()
    emitted: list[str | None] = []
    rt._emit = lambda ev: emitted.append(ev.get("image"))
    old = Image.new("RGB", (12, 12), (200, 10, 10))
    rt._last_image = old
    rt._last_overlay_kps = np.ones((37, 4), dtype=np.float32)
    rt._driven_keypoints = rt._last_overlay_kps
    rt._last_good_keypoints = rt._last_overlay_kps
    rt._inbetween_prev = old
    rt._inbetween_prev_kps = rt._last_overlay_kps
    rt._prev_stream_kps = rt._last_overlay_kps
    rt._prev_stream_hair = np.zeros((1, 4, 4), dtype=np.float32)
    rt._ema_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rt._hair_rig = object()
    rt._hair_capture_done = True
    rt._last_lab_hair = [{}]
    rt._lab_rest_hair = [{}]
    rt._mesh_edited = True
    rt._pose_frozen = True
    rt._freeze_auto_kps = rt._last_overlay_kps
    rt._freeze_params = {}
    rt._freeze_roll = 1.0
    rt._drag_kp_idx = 1
    rt._drag_slots = {1}
    rt._drag_base_kps = rt._last_overlay_kps
    rt._drag_xy = (1.0, 1.0)
    rt._lab_image_wh = (12, 12)
    StreamRuntime._reset_character_runtime(rt, emit_blank=True)
    assert rt._last_image is None
    assert rt._last_overlay_kps is None
    assert rt._inbetween_prev is None
    assert rt._ema_frame is None
    assert rt._hair_rig is None
    assert emitted and emitted[0] is not None
    from backend.stream import _image_to_jpeg_b64

    old_jpeg = _image_to_jpeg_b64(old)
    assert emitted[0] != old_jpeg
    assert rt._lab_image_wh is None


def test_cel_still_squares_nonsquare_preview() -> None:
    from types import SimpleNamespace

    from PIL import Image

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(image_size=32)
    img = Image.new("RGB", (40, 20), (20, 180, 30))
    out = StreamRuntime._cel_still(rt, img, skip_crop=True, image_size=32)
    assert out.size == (32, 32)


def test_cel_still_keeps_matching_square() -> None:
    from types import SimpleNamespace

    from PIL import Image

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt.engine = SimpleNamespace(image_size=32)
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    out = StreamRuntime._cel_still(rt, img, skip_crop=False, image_size=32)
    assert out.size == (32, 32)
    assert list(out.getpixel((0, 0))) == [10, 20, 30]


def test_delete_character_pack_removes_still_and_keys(tmp_path: Path) -> None:
    from backend.character_pack import delete_character_pack

    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    still = character_still_path("gigi", dest_dir=tmp_path)
    still.parent.mkdir(parents=True, exist_ok=True)
    still.write_bytes(b"png")
    keys = dest.with_name("gigi.keys.json")
    keys.write_text("{}", encoding="utf-8")
    delete_character_pack(dest, dest_dir=tmp_path)
    assert not dest.exists()
    assert not still.exists()
    assert not keys.exists()
    assert not still.parent.exists()


def _remove_runtime(tmp_path: Path, dest: Path):
    import threading
    from types import SimpleNamespace

    from PIL import Image

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._listeners = []
    rt._status = {
        "character_id": dest.stem,
        "character_name": "Gigi",
        "reference_name": "Gigi",
        "reference_path": str(dest),
        "show_mesh": True,
        "show_hair": False,
        "travel_box": {"enabled": False},
        "pose_frozen": True,
        "pose_key_count": 2,
        "ref_ready": True,
    }
    rt.status = lambda: dict(rt._status)
    rt._set_status = lambda **kwargs: rt._status.update(kwargs)
    emitted: list[dict] = []
    rt._emit = lambda ev: emitted.append(ev)
    rt._ref_path = dest
    rt._pose_frozen = True
    rt._pose_keys = [{"delta": [[0, 0]]}]
    rt.engine = SimpleNamespace(
        _ref_keypoints=np.ones((37, 4), dtype=np.float32),
        _ref_latent=object(),
        _ref_path=dest,
        _ref_kps_path=None,
        _ref_face_latent=None,
        _ref_keypoints_model=None,
        _ref_keypoints_session_base=None,
        _ref_rig=object(),
        _last_driven_body=object(),
        clear_last_gen_latent=lambda: setattr(rt.engine, "_last_gen_latent", None),
        _last_gen_latent=object(),
    )
    rt._last_image = Image.new("RGB", (12, 12), (200, 10, 10))
    rt._last_overlay_kps = np.ones((37, 4), dtype=np.float32)
    rt._driven_keypoints = rt._last_overlay_kps
    rt._last_good_keypoints = rt._last_overlay_kps
    rt._inbetween_prev = rt._last_image
    rt._inbetween_prev_kps = rt._last_overlay_kps
    rt._prev_stream_kps = rt._last_overlay_kps
    rt._prev_stream_hair = None
    rt._ema_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rt._hair_rig = object()
    rt._hair_capture_done = True
    rt._last_lab_hair = [{}]
    rt._lab_rest_hair = [{}]
    rt._mesh_edited = True
    rt._freeze_auto_kps = rt._last_overlay_kps
    rt._freeze_params = {}
    rt._freeze_roll = 1.0
    rt._drag_kp_idx = 1
    rt._drag_slots = {1}
    rt._drag_base_kps = rt._last_overlay_kps
    rt._drag_xy = (1.0, 1.0)
    rt._travel_ref_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rt._travel_silhouette = None
    rt._travel_silhouette_tight = None
    rt._live_origin_keypoints = np.ones((37, 4), dtype=np.float32)
    rt.list_characters = lambda: []
    rt._emitted = emitted
    return rt


def test_remove_loaded_character_clears_cel_and_overlay(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    dest = tmp_path / "gigi.vtm"
    write_character_pack(dest, **_tiny_pack_payload())
    still = character_still_path("gigi", dest_dir=tmp_path)
    still.parent.mkdir(parents=True, exist_ok=True)
    still.write_bytes(b"png")
    saved: dict[str, str] = {}
    monkeypatch.setattr("backend.character_pack.characters_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "backend.character_pack.resolve_character_id",
        lambda ident, dest_dir=None: dest,
    )
    monkeypatch.setattr("backend.blendshapes.delete_character_plan", lambda ident: None)
    monkeypatch.setattr(
        "backend.ui_session.save_ui_session",
        lambda **kwargs: saved.update({k: str(v) for k, v in kwargs.items()}),
    )
    rt = _remove_runtime(tmp_path, dest)
    result = StreamRuntime.remove_character(rt, "gigi")
    assert rt._last_image is None
    assert rt._last_overlay_kps is None
    assert rt._ref_path is None
    assert rt.engine._ref_latent is None
    assert rt.engine._ref_keypoints is None
    assert rt._status["character_id"] == ""
    assert rt._status["ref_ready"] is False
    assert saved.get("character_path") == ""
    assert result["status"]["character_id"] == ""
    blank = [ev for ev in rt._emitted if ev.get("type") == "frame"]
    assert blank and blank[-1].get("image") is None
    assert not dest.exists()
    assert not still.exists()


def test_remove_other_character_keeps_desk(tmp_path: Path, monkeypatch) -> None:
    from backend.stream import StreamRuntime

    loaded = tmp_path / "gigi.vtm"
    other = tmp_path / "other.vtm"
    write_character_pack(loaded, **_tiny_pack_payload())
    write_character_pack(other, **_tiny_pack_payload())
    monkeypatch.setattr("backend.character_pack.characters_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "backend.character_pack.resolve_character_id",
        lambda ident, dest_dir=None: other,
    )
    monkeypatch.setattr("backend.blendshapes.delete_character_plan", lambda ident: None)
    monkeypatch.setattr("backend.ui_session.save_ui_session", lambda **kwargs: None)
    rt = _remove_runtime(tmp_path, loaded)
    StreamRuntime.remove_character(rt, "other")
    assert other.exists() is False
    assert loaded.exists()
    assert rt._last_image is not None
    assert rt._last_overlay_kps is not None
    assert rt._ref_path == loaded
    assert rt._status["character_id"] == "gigi"
    assert rt.engine._ref_latent is not None


def test_quiet_when_no_character() -> None:
    import threading
    from types import SimpleNamespace

    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._lock = threading.RLock()
    rt._tracking = False
    rt._streaming = False
    rt._frame_in_flight = False
    rt._gen_busy = False
    rt._last_image = None
    rt._last_overlay_kps = None
    rt._ref_path = None
    rt._status = {}
    rt.status = lambda: dict(rt._status)
    rt.engine = SimpleNamespace(_ref_latent=None)
    rt._current_keypoints = lambda: None
    rt.ensure_model = lambda: (_ for _ in ()).throw(AssertionError("model"))
    rt._set_track_status = lambda **kwargs: (_ for _ in ()).throw(AssertionError("track"))
    rt._reset_live_origin = lambda **kwargs: None
    StreamRuntime.start_tracking(rt)
    StreamRuntime.generate_once(rt)
    StreamRuntime.start_stream(rt)
    StreamRuntime.calibrate_ref(rt)
    assert StreamRuntime.freeze_pose(rt) == {}
    assert StreamRuntime._write_lab_source(rt) is None
    assert StreamRuntime.current_character_path(rt) is None


def test_boot_skips_missing_character(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace

    from backend.character_pack import CharacterPackError
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._boot_lock = threading.Lock()
    rt._lock = threading.RLock()
    rt._boot = {
        "awaiting": "character",
        "suggested": "gone",
        "error": "stale",
        "ready": False,
        "running": False,
        "stages": {},
    }
    rt._ref_path = None
    rt._pose_frozen = False
    rt._pose_keys = [{"delta": 1}]
    rt._status = {"error": "Character not found: gone", "character_id": "gone"}
    rt.engine = SimpleNamespace()
    rt._emit = lambda ev: None
    saved: dict[str, str] = {}
    stages: list[dict] = []
    monkeypatch.setattr(
        "backend.ui_session.load_ui_session",
        lambda: {"character_path": "models/characters/gone.vtm"},
    )
    monkeypatch.setattr(
        "backend.stream.previous_load_target",
        lambda session: ("character", "gone"),
    )
    monkeypatch.setattr(
        "backend.ui_session.save_ui_session",
        lambda **kwargs: saved.update({k: str(v) for k, v in kwargs.items()}),
    )
    rt.load_character = lambda *a, **k: (_ for _ in ()).throw(
        CharacterPackError("Character not found: gone")
    )
    rt._set_boot_stage = lambda key, **kwargs: stages.append(kwargs)
    rt._set_status = lambda **kwargs: rt._status.update(kwargs)
    StreamRuntime._boot_character(rt)
    assert stages[-1]["stage_state"] == "skip"
    assert stages[-1].get("error") == ""
    assert rt._status.get("error") == ""
    assert rt._status.get("character_id") == ""
    assert rt._ref_path is None
    assert saved.get("character_path") == ""
