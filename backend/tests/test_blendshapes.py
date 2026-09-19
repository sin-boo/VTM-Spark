from __future__ import annotations

from pathlib import Path

from backend.blendshapes import (
    apply_current_to_character,
    compatibility,
    fingerprint,
    has_plan,
    load_character_plan,
    load_current,
    normalize_shapes,
    plan_card_fields,
    save_character,
    save_current,
)
from backend.stream import StreamRuntime


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


def test_mismatched_snapshot_is_incompatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0)))
    save_character("Gigi", _shapes(("rest", 8.0)))
    assert compatibility("Gigi")["compatible"] is False


def test_repair_copies_current_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("backend.blendshapes.blendshapes_dir", lambda: tmp_path)
    save_current(_shapes(("rest", 0.0), ("O", 4.0)))
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
    save_current(_shapes(("rest", 0.0)))
    save_character("Gigi", _shapes(("rest", 6.0)))
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


def test_load_character_does_not_replace_lab_shapes_by_default() -> None:
    import inspect

    src = inspect.getsource(StreamRuntime.load_character)
    assert "replace_lab: bool = False" in src
    assert "require_compatible: bool = True" in src
    assert "_cel_still(" in src
