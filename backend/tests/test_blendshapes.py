from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.blendshapes import (
    apply_current_to_character,
    compatibility,
    fingerprint,
    has_plan,
    load_character_plan,
    load_current,
    normalize_shapes,
    plan_card_fields,
    refresh_current_from_lab,
    save_character,
    save_current,
)
from backend.stream import StreamRuntime


@pytest.fixture(autouse=True)
def _no_real_characters(tmp_path: Path, monkeypatch) -> None:
    """save_character writes into a real .vtm of the same name: keep tests off the library."""
    library = tmp_path / "characters"
    monkeypatch.setattr("backend.paths.characters_dir", lambda: library)
    monkeypatch.setattr("backend.character_pack.characters_dir", lambda: library)


def _pts(y: float) -> list[list[float]]:
    return [[float(i), float(y), 1.0] for i in range(28)]


def _shapes(*pairs: tuple[str, float]) -> dict[str, list[list[float]]]:
    return {name: _pts(y) for name, y in pairs}


def test_fingerprint_stable_and_order_independent() -> None:
    a = _shapes(("rest", 0.0), ("A", 2.0))
    b = {"A": _pts(2.0), "rest": _pts(0.0)}
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a)
    assert fingerprint(_shapes(("rest", 1.0), ("A", 2.0))) != fingerprint(a)


def test_normalize_drops_short_rows() -> None:
    raw = {"rest": [[1, 2], [3, 4]], "smile": _pts(1.0)}
    out = normalize_shapes(raw)
    assert "rest" not in out
    assert has_plan(out)
    assert len(out["smile"]) == 28


def test_no_current_plan_is_compatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    info = compatibility("Gigi")
    assert info["compatible"] is True
    assert info["has_current_plan"] is False
    assert plan_card_fields("Gigi")["shapes_compatible"] is True


def test_character_without_snapshot_is_incompatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0), ("smile", 1.0)))
    info = compatibility("Gigi")
    assert info["compatible"] is False
    assert info["has_character_plan"] is False
    assert plan_card_fields("Gigi")["has_shapes"] is False


def test_matching_snapshot_is_compatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    plan = _shapes(("rest", 0.0), ("A", 3.0))
    save_current(plan)
    save_character("Gigi", plan)
    info = compatibility("Gigi")
    assert info["compatible"] is True
    assert info["character_fingerprint"] == info["current_fingerprint"]
    assert (tmp_path / "current.json").is_file()
    assert (tmp_path / "Gigi.json").is_file()


def test_rounding_noise_stays_compatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    plan = _shapes(("rest", 0.0), ("smile", 365.949))
    drifted = _shapes(("rest", 0.0), ("smile", 365.95))
    save_current(plan)
    save_character("hi", drifted)
    info = compatibility("hi")
    assert info["compatible"] is True
    assert info["current_fingerprint"] != info["character_fingerprint"]
    assert fingerprint(plan) != fingerprint(drifted)


def test_authored_nudge_is_incompatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("smile", 10.0)))
    save_character("hi", _shapes(("smile", 10.002)))
    assert compatibility("hi")["compatible"] is False


def test_refresh_ignores_rounding_noise(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("smile", 365.949)))
    before = (tmp_path / "current.json").read_bytes()
    refresh_current_from_lab({"shapes": _shapes(("smile", 365.95))})
    assert (tmp_path / "current.json").read_bytes() == before
    refresh_current_from_lab({"shapes": _shapes(("smile", 400.0))})
    assert load_current()["shapes"]["smile"][0][1] == 400.0


def test_mismatched_snapshot_is_incompatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0), ("smile", 1.0)))
    save_character("Gigi", _shapes(("rest", 0.0), ("smile", 1.5)))
    assert compatibility("Gigi")["compatible"] is False


def _rebased(shapes: dict[str, list[list[float]]], dx: float, dy: float, s: float) -> dict:
    return {
        name: [[p[0] * s + dx, p[1] * s + dy, p[2]] for p in rows]
        for name, rows in shapes.items()
    }


def test_plan_rebased_onto_another_face_stays_compatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    plan = _shapes(("rest", 0.0), ("smile", 1.0), ("A", 2.5))
    save_character("Goblin", plan)
    save_current(_rebased(plan, 40.0, -12.0, 1.7))
    assert compatibility("Goblin")["compatible"] is True
    moved = _rebased(plan, 40.0, -12.0, 1.7)
    moved["A"] = [[p[0], p[1] + 2.0, p[2]] for p in moved["A"]]
    save_current(moved)
    assert compatibility("Goblin")["compatible"] is False



def test_eye_shapes_are_part_of_the_plan(tmp_path: Path, monkeypatch) -> None:
    """Eye open / Eye closed travel with a character like the mouth shapes.
    A rebased eye shape still matches; an edited one does not."""
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    plan = _shapes(("rest", 0.0), ("A", 2.5))
    shut = _pts(0.0)
    shut[12][1] += 3.0
    shut[18][1] += 3.0
    plan["eye_closed"] = shut
    assert "eye_closed" in normalize_shapes(plan)
    save_character("Goblin", plan)
    save_current(_rebased(plan, 40.0, -12.0, 1.7))
    assert compatibility("Goblin")["compatible"] is True
    moved = _rebased(plan, 40.0, -12.0, 1.7)
    moved["eye_closed"][12][1] += 2.0
    save_current(moved)
    assert compatibility("Goblin")["compatible"] is False

def test_lab_resave_drift_stays_compatible(tmp_path: Path, monkeypatch) -> None:
    """A re-saved plan drifts a few 0.001 px steps in the lips; not a new plan."""
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    plan = _shapes(("rest", 0.0), ("smile", 1.0), ("A", 2.5))
    save_character("Drift", plan)
    drifted = _shapes(("rest", 0.0), ("smile", 1.0), ("A", 2.5))
    drifted["A"][23][0] -= 0.003
    drifted["smile"][26][0] += 0.003
    save_current(drifted)
    assert compatibility("Drift")["compatible"] is True
    drifted["A"][23][0] -= 0.5
    save_current(drifted)
    assert compatibility("Drift")["compatible"] is False


def test_repair_copies_current_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0), ("E", 4.0)))
    save_character("Gigi", _shapes(("rest", 9.0)))
    apply_current_to_character("Gigi")
    assert compatibility("Gigi")["compatible"] is True
    assert load_character_plan("Gigi")["fingerprint"] == load_current()["fingerprint"]


def test_create_snapshots_current_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0), ("sad", 1.5)))
    rt = object.__new__(StreamRuntime)
    rt._refresh_blend_current = lambda: None
    StreamRuntime._snapshot_character_shapes(rt, "Gigi")
    assert compatibility("Gigi")["compatible"] is True
    assert load_character_plan("Gigi")["shapes"]["sad"][0][1] == 1.5


def test_shape_gate_blocks_then_repairs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0), ("smile", 1.0)))
    save_character("Gigi", _shapes(("rest", 6.0), ("smile", 9.0)))
    fake = tmp_path / "Gigi.vtm"
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "backend.character_pack.resolve_character_id",
        lambda ident, dest_dir=None: fake,
    )
    rt = object.__new__(StreamRuntime)
    rt._refresh_blend_current = lambda: None
    rt.status = lambda: {"state": "idle"}
    rt._character_card_safe = lambda path: {"id": path.stem, "name": "Gigi"}

    blocked = StreamRuntime._character_shape_gate(
        rt, "Gigi", repair=False, require_compatible=True
    )
    assert blocked is not None
    assert blocked["incompatible"] is True
    assert "repair" in blocked["message"].lower()
    assert compatibility("Gigi")["compatible"] is False

    opened = StreamRuntime._character_shape_gate(
        rt, "Gigi", repair=True, require_compatible=True
    )
    assert opened is None
    assert compatibility("Gigi")["compatible"] is True


def test_apply_reference_loads_vtm_instead_of_opening_it(tmp_path: Path, monkeypatch) -> None:
    pack = tmp_path / "hi.vtm"
    pack.write_bytes(b"not an image")
    monkeypatch.setattr("backend.stream.resolve_user_path", lambda _path: pack)
    rt = object.__new__(StreamRuntime)
    seen: dict[str, str] = {}

    def load(ident: str, **_kwargs: object) -> dict[str, object]:
        seen["ident"] = ident
        return {"frame": {"type": "frame", "image": "ok"}}

    rt.load_character = load
    frame = StreamRuntime.apply_reference(rt, pack)
    assert seen["ident"] == "hi"
    assert frame["type"] == "frame"


def test_socket_replays_the_loaded_still() -> None:
    from pathlib import Path

    from PIL import Image

    rt = object.__new__(StreamRuntime)
    rt._last_image = Image.new("RGB", (4, 4), (9, 8, 7))
    rt._last_overlay_kps = None
    rt._frame_payload = lambda image, _kps: {
        "image": "still",
        "width": image.size[0],
        "height": image.size[1],
    }
    event = StreamRuntime.current_frame_event(rt)
    assert event is not None
    assert event["type"] == "frame"
    assert event["image"] == "still"
    api = Path(__file__).resolve().parents[1].joinpath("api.py").read_text(encoding="utf-8")
    assert "current_frame_event()" in api


def test_load_character_does_not_replace_lab_shapes_by_default() -> None:
    import inspect

    src = inspect.getsource(StreamRuntime.load_character)
    assert "replace_lab: bool = False" in src
    assert "require_compatible: bool = True" in src
    assert "return blocked" not in src
    assert "_cel_still(" in src


def test_load_character_repairs_after_lab_holds_the_still() -> None:
    import inspect

    src = inspect.getsource(StreamRuntime.load_character)
    assert src.index("_sync_lab_character(") < src.index("_character_shape_gate(")


def test_start_tracking_restores_painted_fit() -> None:
    import inspect

    src = inspect.getsource(StreamRuntime._start_lab_tracking)
    assert src.index("_sync_lab_character()") < src.index("_restore_character_fit()")
    assert "_restore_character_fit()" in inspect.getsource(StreamRuntime._boot_lab_source)


def test_model_switch_skips_opening_vtm_then_loads_character(tmp_path: Path, monkeypatch) -> None:
    import threading

    from backend.engine import StreamEngine

    pack = tmp_path / "ChatGPT-Image.vtm"
    pack.write_bytes(b"not an image")
    ckpt = tmp_path / "next.pt"
    ckpt.write_bytes(b"weights")
    previous = tmp_path / "old.pt"
    previous.write_bytes(b"old")

    eng = object.__new__(StreamEngine)
    eng._cuda_lock = threading.Lock()
    eng.checkpoint = previous
    eng._ready = True
    eng.model = object()
    eng._ref_path = pack
    eng._ref_keypoints = object()
    eng._compile_failed = False
    opened: list[str] = []
    eng._release_dit_weights = lambda: None
    eng.load = lambda **_k: None
    eng._set_reference_locked = lambda *_a, **_k: opened.append("image")
    assert StreamEngine.set_checkpoint(eng, ckpt) == ckpt
    assert opened == []

    order: list[str] = []
    rt = object.__new__(StreamRuntime)
    rt._ref_path = pack
    rt._model_load_lock = threading.Lock()
    rt._lock = threading.RLock()
    rt._fast_warmed = True
    rt._batch2_auto_tried = True
    rt._set_status = lambda **_k: None
    rt._clear_progress = lambda **_k: order.append("ready")

    class _Meter:
        def __enter__(self) -> "_Meter":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def stage(self, _key: str, _label: str) -> None:
            return None

    rt._stage_meter = lambda *_a, **_k: _Meter()

    class _Engine:
        checkpoint = ckpt
        image_size = 768
        vae = object()

        def set_checkpoint(self, _path: Path, on_stage: object = None) -> None:
            order.append("model")

        def set_stream_batch_size(self, _n: int) -> None:
            return None

        def load_encoded_reference(self, **_kwargs: object) -> None:
            order.append("character")

        def model_identity(self) -> dict[str, object]:
            return {"checkpoint": "next.pt", "image_size": 768, "latent_shape": [1, 4, 96, 96]}

    rt.engine = _Engine()
    monkeypatch.setattr("backend.stream.resolve_user_path", lambda path: Path(path))
    monkeypatch.setattr("backend.stream.is_stream_checkpoint_file", lambda _path: True)
    monkeypatch.setattr("backend.stream.remember_checkpoint_location", lambda _path: None)
    monkeypatch.setattr("backend.stream.checkpoint_label", lambda _path: "next")
    monkeypatch.setattr("backend.stream.display_path", lambda path: str(path))
    monkeypatch.setattr("backend.ui_session.save_ui_session", lambda **_k: None)
    monkeypatch.setattr(
        "backend.character_pack.read_character_pack",
        lambda _path: type(
            "Pack",
            (),
            {
                "image_size": 768,
                "keypoints": None,
                "ref_latent": np.zeros((1, 4, 96, 96), dtype=np.float32),
                "ref_face_latent": None,
                "skip_crop": True,
                "model": {"checkpoint": "next.pt", "image_size": 768},
            },
        )(),
    )

    StreamRuntime.set_checkpoint(rt, ckpt)
    assert order == ["model", "character", "ready"]
