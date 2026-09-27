"""Track Lab plan / blend shapes: the desk's working plan and each character's.

- ``models/blendshapes/current.json`` — latest plan copied from Track Lab
  (read-only for the lab); stays on this machine.
- A character's snapshot (taken when it was created or last repaired) lives in
  its ``.vtm`` as ``blendshapes.json`` so it travels with the pack. Older
  installs kept it as ``models/blendshapes/<character-id>.json``; the first
  read folds that file into the pack and deletes it. Without a pack the
  ``<character-id>.json`` file is still used.

A plan made on this desk has ``origin: "desk"``; a plan that arrived inside an
imported pack is marked ``origin: "imported"``. An imported plan is the
creator's and is never reported as needing repair just because this desk's
lab plan differs (see :func:`compatibility`).

VTM Noble reads lab shapes and writes these files. It does not write shapes
back into Track Lab.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import blendshapes_dir, display_path

FORMAT_ID = "vtm-blendshapes"
FORMAT_VERSION = 1
CURRENT_ID = "current"
ORIGIN_DESK = "desk"
ORIGIN_IMPORTED = "imported"
SHAPE_IDS = ("rest", "smile", "sad", "A", "I", "U", "E")  # same list as Track Lab's authored shapes (track_lab/backend/presets.py PRESET_IDS)
INCOMPATIBLE_MESSAGE = (
    "Incompatible. This character's blend shapes do not match the current plan. "
    "Would you like us to repair this character?"
)


def current_plan_path() -> Path:
    return blendshapes_dir() / "current.json"


def character_plan_path(ident: str) -> Path:
    stem = Path(str(ident or "").strip()).name
    if not stem or stem in {".", ".."}:
        raise ValueError("Invalid character id")
    return blendshapes_dir() / f"{stem}.json"


def normalize_shapes(raw: object) -> dict[str, list[list[float]]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[list[float]]] = {}
    for name in SHAPE_IDS:
        pts = raw.get(name)
        if not isinstance(pts, list) or len(pts) < 28:
            continue
        rows: list[list[float]] = []
        for row in pts[:28]:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                rows.append([0.0, 0.0, 0.0])
                continue
            score = float(row[2]) if len(row) > 2 else 1.0
            rows.append([round(float(row[0]), 3), round(float(row[1]), 3), round(score, 4)])
        out[name] = rows
    return out


def has_plan(shapes: object) -> bool:
    if not isinstance(shapes, dict):
        return False
    return any(name in shapes for name in SHAPE_IDS)


def fingerprint(shapes: object) -> str:
    norm = normalize_shapes(shapes)
    if not has_plan(norm):
        return ""
    payload = json.dumps(norm, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# One stored step after normalize (xy to 0.001, score to 0.0001). Wider than that is a new plan.
SHAPE_MATCH_XY = 0.001
SHAPE_MATCH_SCORE = 0.0001


def shapes_match(a: object, b: object) -> bool:
    """True when two plans are the same after rounding, including one-step noise."""
    left = normalize_shapes(a)
    right = normalize_shapes(b)
    if set(left) != set(right):
        return False
    for name, rows in left.items():
        other = right[name]
        if len(rows) != len(other):
            return False
        for p, q in zip(rows, other):
            if abs(p[0] - q[0]) > SHAPE_MATCH_XY or abs(p[1] - q[1]) > SHAPE_MATCH_XY:
                return False
            if abs(p[2] - q[2]) > SHAPE_MATCH_SCORE:
                return False
    return True


MOUTH_SLOTS = tuple(range(20, 28))
_MOUTH_RIGHT = 23
_MOUTH_LEFT = 26


def _mouth_offsets(
    shapes: dict[str, list[list[float]]],
) -> tuple[dict[str, list[tuple[float, float]]], float] | None:
    """Each shape's lips relative to rest, in rest's mouth frame, per mouth width."""
    rest = shapes.get("rest")
    if not rest:
        return None
    rx, ry = rest[_MOUTH_RIGHT][0], rest[_MOUTH_RIGHT][1]
    ax, ay = rest[_MOUTH_LEFT][0] - rx, rest[_MOUTH_LEFT][1] - ry
    width = (ax * ax + ay * ay) ** 0.5
    if width < 1e-3:
        return None
    ax, ay = ax / width, ay / width
    dx, dy = -ay, ax
    if dy < 0.0:
        dx, dy = -dx, -dy
    out: dict[str, list[tuple[float, float]]] = {}
    for name, rows in shapes.items():
        if name == "rest":
            continue
        offs: list[tuple[float, float]] = []
        for slot in MOUTH_SLOTS:
            ox = rows[slot][0] - rest[slot][0]
            oy = rows[slot][1] - rest[slot][1]
            offs.append(((ox * ax + oy * ay) / width, (ox * dx + oy * dy) / width))
        out[name] = offs
    return out, width


def plans_match(a: object, b: object) -> bool:
    """Same authored visemes, wherever the lab's rest face currently sits.

    Track Lab rebases every shape onto the rest of the still it has loaded, so
    raw pixels move whenever a different character is tracked. The lip offsets
    from rest, in the mouth's own frame and per mouth width, are what that
    rebase keeps — compare those. Without a rest, fall back to raw pixels.
    """
    left = normalize_shapes(a)
    right = normalize_shapes(b)
    if set(left) != set(right):
        return False
    fa = _mouth_offsets(left)
    fb = _mouth_offsets(right)
    if fa is None or fb is None:
        return shapes_match(left, right)
    offs_a, width_a = fa
    offs_b, width_b = fb
    tol = 2.5 * SHAPE_MATCH_XY / min(width_a, width_b)
    for name, rows in offs_a.items():
        for p, q in zip(rows, offs_b[name]):
            if abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol:
                return False
    return True


def empty_plan(ident: str = CURRENT_ID) -> dict[str, Any]:
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "id": ident,
        "fingerprint": "",
        "shapes": {},
    }


def plan_payload(shapes: object, *, ident: str, origin: str = "") -> dict[str, Any]:
    norm = normalize_shapes(shapes)
    payload: dict[str, Any] = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "id": str(ident),
        "fingerprint": fingerprint(norm),
        "shapes": norm,
    }
    if origin:
        payload["origin"] = str(origin)
    return payload


def _plan_from_raw(raw: object, ident: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return empty_plan(ident)
    shapes = normalize_shapes(raw.get("shapes"))
    out: dict[str, Any] = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "id": str(raw.get("id") or ident),
        # Recomputed, not read back: a plan saved with a since-dropped shape (O)
        # carries a fingerprint over shapes it no longer has.
        "fingerprint": fingerprint(shapes),
        "shapes": shapes,
    }
    origin = str(raw.get("origin") or "")
    if origin:
        out["origin"] = origin
    return out


def write_plan(
    path: Path | str, shapes: object, *, ident: str, origin: str = ""
) -> dict[str, Any]:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = plan_payload(shapes, ident=str(ident or dest.stem), origin=origin)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def read_plan(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return empty_plan(file_path.stem)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_plan(file_path.stem)
    return _plan_from_raw(raw, file_path.stem)


def character_pack_path(ident: str) -> Path | None:
    """The character's ``.vtm`` by exact id (no name search), or None."""
    from .paths import characters_dir

    stem = Path(str(ident or "").strip()).name
    if not stem or stem in {".", ".."}:
        return None
    folder = characters_dir()
    for cand in (folder / f"{stem}.vtm", folder / stem / f"{stem}.vtm"):
        if cand.is_file():
            return cand
    return None


def _fold_legacy_plan(ident: str, pack: Path) -> None:
    """Move ``models/blendshapes/<id>.json`` into the pack once, then delete it.

    A plan already inside the pack wins; the file only fills an empty one.
    """
    from .character_pack import CharacterPackError, upgrade_character_pack

    legacy = character_plan_path(ident)
    if not legacy.is_file():
        return
    plan = read_plan(legacy)
    if has_plan(plan.get("shapes")):
        payload = plan_payload(
            plan["shapes"],
            ident=pack.stem,
            origin=str(plan.get("origin") or ORIGIN_DESK),
        )
        try:
            upgrade_character_pack(pack, blendshapes=payload)
        except (CharacterPackError, OSError):
            return  # keep the file; the pack could not take it
    try:
        legacy.unlink()
    except OSError:
        pass


def load_current() -> dict[str, Any]:
    return read_plan(current_plan_path())


def load_character_plan(ident: str) -> dict[str, Any]:
    from .character_pack import CharacterPackError, read_pack_blendshapes

    pack = character_pack_path(ident)
    if pack is None:
        return read_plan(character_plan_path(ident))
    _fold_legacy_plan(ident, pack)
    try:
        raw = read_pack_blendshapes(pack)
    except (CharacterPackError, OSError):
        return read_plan(character_plan_path(ident))
    return _plan_from_raw(raw, pack.stem) if raw else empty_plan(pack.stem)


def save_current(shapes: object) -> dict[str, Any]:
    return write_plan(current_plan_path(), shapes, ident=CURRENT_ID)


def save_character(ident: str, shapes: object, *, origin: str = ORIGIN_DESK) -> dict[str, Any]:
    """Store a character's plan: inside its ``.vtm`` when there is one."""
    from .character_pack import write_pack_blendshapes

    pack = character_pack_path(ident)
    if pack is None:
        return write_plan(character_plan_path(ident), shapes, ident=ident, origin=origin)
    payload = plan_payload(shapes, ident=pack.stem, origin=origin)
    write_pack_blendshapes(pack, payload)
    legacy = character_plan_path(ident)
    if legacy.is_file():
        legacy.unlink()
    return payload


def mark_plan_imported(pack: Path | str) -> bool:
    """An imported pack's plan is the creator's: tag it so repair never replaces it."""
    from .character_pack import read_pack_blendshapes, write_pack_blendshapes

    path = Path(pack)
    raw = read_pack_blendshapes(path)
    plan = _plan_from_raw(raw, path.stem) if raw else empty_plan(path.stem)
    if not has_plan(plan.get("shapes")) or plan.get("origin") == ORIGIN_IMPORTED:
        return False
    write_pack_blendshapes(
        path, plan_payload(plan["shapes"], ident=path.stem, origin=ORIGIN_IMPORTED)
    )
    return True


def delete_character_plan(ident: str) -> None:
    """Drop the legacy plan file. A pack's own plan goes with the pack."""
    path = character_plan_path(ident)
    if path.is_file():
        path.unlink()


def shapes_from_lab(packet: dict[str, Any] | None) -> dict[str, list[list[float]]]:
    if not isinstance(packet, dict):
        return {}
    return normalize_shapes(packet.get("shapes"))


def refresh_current_from_lab(packet: dict[str, Any] | None) -> dict[str, Any]:
    """Copy authored lab shapes into ``current.json``. Never writes to the lab.

    Rounding noise from the lab is not a new plan, so an unchanged authoring
    pass does not rewrite the file or invalidate characters created from it.
    """
    shapes = shapes_from_lab(packet)
    if not has_plan(shapes):
        return load_current()
    current = load_current()
    if has_plan(current.get("shapes")) and shapes_match(current.get("shapes"), shapes):
        return current
    return save_current(shapes)


def apply_current_to_character(ident: str) -> dict[str, Any]:
    current = load_current()
    if not has_plan(current.get("shapes")):
        raise ValueError("No current blend shapes to repair this character with")
    mine = load_character_plan(ident)
    if has_plan(mine.get("shapes")) and mine.get("origin") == ORIGIN_IMPORTED:
        raise ValueError("This character's blend shapes came with its pack and are kept")
    return save_character(ident, current["shapes"])


def compatibility(ident: str, current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare a character snapshot to the current plan.

    Repair is offered only when this desk's lab plan has moved on from a plan
    the character took from this desk, or the character has none while the
    desk has one. A plan that came inside an imported pack is the creator's:
    a different (or missing) plan here does not break it and repair must not
    replace it. ``matches_current`` still reports whether the two agree.
    """
    plan = current if isinstance(current, dict) else load_current()
    cur_shapes = plan.get("shapes") if isinstance(plan.get("shapes"), dict) else {}
    char = load_character_plan(ident)
    char_shapes = char.get("shapes") if isinstance(char.get("shapes"), dict) else {}
    current_fp = fingerprint(cur_shapes)
    character_fp = fingerprint(char_shapes)
    has_current = bool(current_fp)
    has_character = bool(character_fp)
    imported = has_character and char.get("origin") == ORIGIN_IMPORTED
    matches = has_current and has_character and plans_match(cur_shapes, char_shapes)
    compatible = (not has_current) or imported or matches
    pack = character_pack_path(ident)
    return {
        "compatible": bool(compatible),
        "has_current_plan": has_current,
        "has_character_plan": has_character,
        "imported_plan": bool(imported),
        "matches_current": bool(matches),
        "current_fingerprint": current_fp,
        "character_fingerprint": character_fp,
        "path": display_path(pack if pack is not None else character_plan_path(ident)),
        "current_path": display_path(current_plan_path()),
    }


def plan_card_fields(ident: str, current: dict[str, Any] | None = None) -> dict[str, Any]:
    info = compatibility(ident, current)
    return {
        "shapes_compatible": bool(info["compatible"]),
        "has_shapes": bool(info["has_character_plan"]),
        "shapes_path": str(info["path"]),
    }
