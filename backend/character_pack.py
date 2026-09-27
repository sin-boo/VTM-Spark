"""Portable VTM character packs (``.vtm`` = zip, no compile artifacts).

A pack is the whole character another NVIDIA box can load without VAE-encoding
the still again and without any sidecar files:

* ``manifest.json``   identity, model it was encoded for, creator meta, and a
  sha256 for every other member (format v2)
* ``preview.png``     the cel still; ``thumb.png`` a small copy for the library
* ``source.<ext>``    optional original upload so another model can re-encode
* ``latents.npz``     float16 reference latents; ``keypoints.npy`` rest pose
* ``fit.json``        painted hair mask, fitted skeleton, limiters / travel box
* ``pose_keys.json``  ``{"keys": [...]}`` as :mod:`pose_keys` saves them
* ``blendshapes.json`` the character's blendshape plan

Readers accept v1 packs (no hashes, no v2 members). Every writer rewrites the
file atomically as v2.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
from PIL import Image

from .paths import characters_dir, display_path

FORMAT_ID = "vtm-character"
FORMAT_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
APP_NAME = "VTM Studio"

MANIFEST_NAME = "manifest.json"
PREVIEW_NAME = "preview.png"
THUMB_NAME = "thumb.png"
SOURCE_NAME_PREFIX = "source"
LATENTS_NAME = "latents.npz"
KEYPOINTS_NAME = "keypoints.npy"
FIT_NAME = "fit.json"
POSE_KEYS_NAME = "pose_keys.json"
BLENDSHAPES_NAME = "blendshapes.json"

THUMB_MAX_SIDE = 256
MAX_PACK_BYTES = 512 * 1024 * 1024
MAX_PACK_MEMBERS = 256
META_LIMITS = {"author": 120, "license": 120, "description": 2000}
SOURCE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

_KNOWN_MEMBERS = (
    PREVIEW_NAME,
    THUMB_NAME,
    LATENTS_NAME,
    KEYPOINTS_NAME,
    FIT_NAME,
    POSE_KEYS_NAME,
    BLENDSHAPES_NAME,
)
_STORED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_locks_guard = threading.Lock()
_path_locks: dict[str, threading.RLock] = {}


class CharacterPackError(ValueError):
    """``.vtm`` is missing, corrupt, or not a character pack."""


@dataclass
class CharacterPack:
    name: str
    preview_rgb: np.ndarray
    keypoints: np.ndarray
    ref_latent: np.ndarray
    ref_face_latent: np.ndarray | None
    image_size: int
    skip_crop: bool
    source_name: str
    has_face_latent: bool
    fit: dict[str, Any] = field(default_factory=dict)
    version: int = FORMAT_VERSION
    pose_keys: list[Any] = field(default_factory=list)
    blendshapes: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    source_bytes: bytes | None = None
    source_suffix: str = ".png"

    def manifest(self) -> dict[str, Any]:
        """v2 manifest for this pack (without the member hashes)."""
        return {
            "format": FORMAT_ID,
            "version": FORMAT_VERSION,
            "name": self.name,
            "image_size": int(self.image_size),
            "skip_crop": bool(self.skip_crop),
            "has_face_latent": bool(self.ref_face_latent is not None),
            "source_name": self.source_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "app": _app_info(),
            "model": _model_info(
                self.model,
                image_size=self.image_size,
                latent_shape=np.shape(self.ref_latent),
            ),
            "meta": sanitize_pack_meta(self.meta),
        }


# ---------------------------------------------------------------- small helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_from_timestamp(stamp: float) -> str:
    moment = datetime.fromtimestamp(stamp, tz=timezone.utc)
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _app_info() -> dict[str, Any]:
    return {"name": APP_NAME, "pack_version": FORMAT_VERSION}


def _file_name_only(value: Any) -> str:
    """Only the file name: a shared pack must not leak the creator's folders."""
    text = str(value or "").strip()
    return re.split(r"[\\/]", text)[-1] if text else ""


def _model_info(
    model: dict[str, Any] | None,
    *,
    image_size: int,
    latent_shape: Iterable[int] | None,
) -> dict[str, Any]:
    src = model if isinstance(model, dict) else {}
    shape = src.get("latent_shape") if latent_shape is None else latent_shape
    try:
        # The pack's own image_size is what the latents were encoded at.
        size = int(image_size or src.get("image_size") or 0)
    except (TypeError, ValueError):
        size = 0
    try:
        dims = [int(v) for v in (shape or [])]
    except (TypeError, ValueError):
        dims = []
    return {
        "checkpoint": _file_name_only(src.get("checkpoint")),
        "image_size": size,
        "latent_shape": dims,
        "latent_dtype": "float16",
    }


def _clean_text(value: Any, limit: int, *, multiline: bool) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    if not multiline:
        text = " ".join(text.split())
    return text.strip()[:limit].strip()


def sanitize_pack_meta(meta: dict[str, Any] | None) -> dict[str, str]:
    """Creator info kept in the manifest: strip, drop control chars, cap length."""
    src = meta if isinstance(meta, dict) else {}
    return {
        key: _clean_text(src.get(key), limit, multiline=key == "description")
        for key, limit in META_LIMITS.items()
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json_bytes(data: Any) -> bytes:
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
    except (TypeError, ValueError) as exc:
        raise CharacterPackError(f"Character data cannot be saved: {exc}") from exc
    return text.encode("utf-8")


def _is_safe_member_name(name: str) -> bool:
    if not name or name != name.strip() or len(name) > 128:
        return False
    if "/" in name or "\\" in name or ":" in name or ".." in name:
        return False
    return not _CONTROL_RE.search(name) and "\n" not in name and "\t" not in name


def _is_source_member(name: str) -> bool:
    stem, dot, suffix = name.partition(".")
    return stem == SOURCE_NAME_PREFIX and bool(dot) and f".{suffix.lower()}" in SOURCE_SUFFIXES


def _source_suffix(suffix: str) -> str:
    ext = str(suffix or ".png").strip().lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext if ext in SOURCE_SUFFIXES else ".png"


def _is_known_member(name: str) -> bool:
    return name in _KNOWN_MEMBERS or _is_source_member(name)


def _compress_type(name: str) -> int:
    return zipfile.ZIP_STORED if Path(name).suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED


def _member_order(name: str) -> tuple[int, str]:
    if name in _KNOWN_MEMBERS:
        return (_KNOWN_MEMBERS.index(name), name)
    if _is_source_member(name):
        return (len(_KNOWN_MEMBERS), name)
    return (len(_KNOWN_MEMBERS) + 1, name)


def _path_lock(path: Path) -> threading.RLock:
    try:
        key = str(path.resolve()).lower()
    except OSError:
        key = str(path).lower()
    with _locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = _path_locks[key] = threading.RLock()
        return lock


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _thumb_from_image(img: Image.Image) -> bytes:
    thumb = img.convert("RGB")
    thumb.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _thumb_from_png(preview_png: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(preview_png)) as img:
            return _thumb_from_image(img)
    except Exception as exc:
        raise CharacterPackError("Character pack preview is unreadable") from exc


def _latent_shape_from_npz(data: bytes) -> list[int]:
    try:
        with np.load(io.BytesIO(data)) as latents:
            return [int(v) for v in latents["ref_latent"].shape]
    except Exception as exc:
        raise CharacterPackError("Character pack latents are unreadable") from exc


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Temp file in the same folder, fsync, then ``os.replace``. Never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows: a reader / scanner may hold the target for a moment.
                if attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ------------------------------------------------------------------ zip reading


def _read_zip(path: Path) -> zipfile.ZipFile:
    """Open a pack and refuse hostile layouts before anything is read."""
    try:
        zf = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise CharacterPackError(f"Not a character pack: {path.name}") from exc
    try:
        infos = zf.infolist()
        if len(infos) > MAX_PACK_MEMBERS:
            raise CharacterPackError("Character pack has too many files inside")
        seen: set[str] = set()
        total = 0
        for info in infos:
            if not _is_safe_member_name(info.filename) or info.is_dir():
                raise CharacterPackError("Character pack has an unsafe file name inside")
            if info.filename in seen:
                raise CharacterPackError(f"Character pack has a duplicate entry: {info.filename}")
            seen.add(info.filename)
            total += max(int(info.file_size), 0)
        if total > MAX_PACK_BYTES:
            raise CharacterPackError("Character pack is too large to open")
    except BaseException:
        zf.close()
        raise
    return zf


def peek_character_manifest(path: Path | str) -> dict[str, Any]:
    zf = _read_zip(Path(path))
    try:
        return _load_manifest(zf)
    finally:
        zf.close()


def _load_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        raw = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CharacterPackError("Character pack is missing a valid manifest") from exc
    if not isinstance(raw, dict) or raw.get("format") != FORMAT_ID:
        raise CharacterPackError("File is not a VTM character pack")
    try:
        version = int(raw.get("version", 0))
    except (TypeError, ValueError) as exc:
        raise CharacterPackError("Character pack version is invalid") from exc
    if version > FORMAT_VERSION:
        raise CharacterPackError(
            f"Character pack version {version} needs a newer {APP_NAME}"
        )
    if version not in SUPPORTED_VERSIONS:
        raise CharacterPackError(f"Unsupported character pack version {version}")
    if version >= 2 and not isinstance(raw.get("members"), dict):
        raise CharacterPackError("Character pack is corrupt: manifest.json")
    return raw


def _manifest_version(raw: dict[str, Any]) -> int:
    return int(raw.get("version", 1))


def _read_member(
    zf: zipfile.ZipFile, raw: dict[str, Any], name: str
) -> bytes | None:
    """One member's bytes (``None`` if absent). v2 hashes are checked."""
    hashes = raw.get("members") if _manifest_version(raw) >= 2 else None
    try:
        data = zf.read(name)
    except KeyError:
        if hashes is not None and name in hashes:
            raise CharacterPackError(f"Character pack is corrupt: {name}") from None
        return None
    except Exception as exc:  # BadZipFile (CRC), zlib.error, truncated file
        raise CharacterPackError(f"Character pack is corrupt: {name}") from exc
    if hashes is not None:
        want = hashes.get(name)
        if not isinstance(want, str) or want.lower() != _sha256(data):
            raise CharacterPackError(f"Character pack is corrupt: {name}")
    return data


def _load_members(
    zf: zipfile.ZipFile, raw: dict[str, Any], *, known_only: bool = False
) -> dict[str, bytes]:
    """Every member except the manifest, hash-checked. Missing listed ones are corrupt.

    Unknown members are skipped unless a v2 manifest lists them (a newer app's
    data survives edits; stray junk is dropped on the next rewrite).
    """
    listed = raw.get("members") if _manifest_version(raw) >= 2 else {}
    out: dict[str, bytes] = {}
    for name in zf.namelist():
        if name == MANIFEST_NAME:
            continue
        if not _is_known_member(name) and (known_only or name not in listed):
            continue
        data = _read_member(zf, raw, name)
        if data is not None:
            out[name] = data
    if _manifest_version(raw) >= 2:
        for name in raw.get("members") or {}:
            if name not in out and (not known_only or _is_known_member(name)):
                raise CharacterPackError(f"Character pack is corrupt: {name}")
    return out


def _parse_json(data: bytes | None, default: Any) -> Any:
    if data is None:
        return default
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return default
    if isinstance(default, dict) and not isinstance(value, dict):
        return default
    if isinstance(default, list) and not isinstance(value, list):
        return default
    return value


def _pose_keys_from_json(value: Any) -> list[Any]:
    items = value.get("keys") if isinstance(value, dict) else value
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _source_member(names: Iterable[str]) -> str | None:
    found = sorted(name for name in names if _is_source_member(name))
    return found[0] if found else None


def _read_fit(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Packs written before fit.json existed simply have none."""
    raw = _load_manifest(zf)
    return _parse_json(_read_member(zf, raw, FIT_NAME), {})


# ------------------------------------------------------------------ zip writing


def _build_pack_bytes(manifest: dict[str, Any], members: dict[str, bytes]) -> tuple[bytes, dict[str, Any]]:
    """Zip ``members`` under a v2 manifest carrying their sha256 hashes."""
    for name in members:
        if name == MANIFEST_NAME or not _is_safe_member_name(name):
            raise CharacterPackError(f"Invalid character pack entry name: {name!r}")
    for required in (PREVIEW_NAME, LATENTS_NAME, KEYPOINTS_NAME):
        if required not in members:
            raise CharacterPackError(f"Character pack is missing {required}")
    total = sum(len(data) for data in members.values())
    if total > MAX_PACK_BYTES:
        raise CharacterPackError("Character pack is too large to save")
    ordered = sorted(members, key=_member_order)
    out = dict(manifest)
    out["format"] = FORMAT_ID
    out["version"] = FORMAT_VERSION
    out["app"] = _app_info()
    out["meta"] = sanitize_pack_meta(out.get("meta"))
    out["members"] = {name: _sha256(members[name]) for name in ordered}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, _json_bytes(out), compress_type=zipfile.ZIP_DEFLATED)
        for name in ordered:
            zf.writestr(name, members[name], compress_type=_compress_type(name))
    return buf.getvalue(), out


def _upgrade_manifest(
    raw: dict[str, Any],
    members: dict[str, bytes],
    *,
    fallback_stamp: str,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """v2 manifest fields from a v1 or v2 manifest; hashes are added on write."""
    out = {
        key: raw[key]
        for key in raw
        if key not in {"format", "version", "app", "members"}
    }
    out["name"] = str(raw.get("name") or "")
    try:
        out["image_size"] = int(raw.get("image_size") or 0)
    except (TypeError, ValueError):
        out["image_size"] = 0
    out["skip_crop"] = bool(raw.get("skip_crop"))
    out["has_face_latent"] = bool(raw.get("has_face_latent"))
    out["source_name"] = str(raw.get("source_name") or "")
    out["created_at"] = str(raw.get("created_at") or fallback_stamp)
    out["updated_at"] = str(raw.get("updated_at") or out["created_at"])
    base_model = dict(raw["model"]) if isinstance(raw.get("model"), dict) else {}
    if isinstance(model, dict):
        for key, value in model.items():
            # Only fill what the pack does not know; never overwrite it.
            if value not in (None, "", [], 0) and base_model.get(key) in (None, "", [], 0):
                base_model[key] = value
    shape = None
    if LATENTS_NAME in members:
        shape = _latent_shape_from_npz(members[LATENTS_NAME])
    out["model"] = _model_info(base_model, image_size=out["image_size"], latent_shape=shape)
    out["meta"] = sanitize_pack_meta(raw.get("meta"))
    return out


def _file_stamp(path: Path) -> str:
    try:
        return _utc_from_timestamp(path.stat().st_mtime)
    except OSError:
        return _utc_now()


def _load_for_rewrite(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    zf = _read_zip(path)
    try:
        raw = _load_manifest(zf)
        members = _load_members(zf, raw)
    finally:
        zf.close()
    return raw, members


def _rewrite_pack(
    path: Path | str,
    *,
    members: dict[str, bytes] | None = None,
    manifest_patch: dict[str, Any] | None = None,
    touch: bool = True,
) -> dict[str, Any]:
    """Read, patch, and atomically rewrite a pack as v2. Returns the new manifest."""
    file_path = Path(path)
    with _path_lock(file_path):
        raw, current = _load_for_rewrite(file_path)
        manifest = _upgrade_manifest(raw, current, fallback_stamp=_file_stamp(file_path))
        new_members = dict(members or {})
        for name in new_members:
            if name == MANIFEST_NAME or not _is_safe_member_name(name):
                raise CharacterPackError(f"Invalid character pack entry name: {name!r}")
            if _is_source_member(name):
                for old in [n for n in current if _is_source_member(n)]:
                    del current[old]
        current.update(new_members)
        if PREVIEW_NAME in new_members and THUMB_NAME not in new_members:
            current[THUMB_NAME] = _thumb_from_png(current[PREVIEW_NAME])
        elif THUMB_NAME not in current and PREVIEW_NAME in current:
            current[THUMB_NAME] = _thumb_from_png(current[PREVIEW_NAME])
        if LATENTS_NAME in new_members:
            manifest["model"]["latent_shape"] = _latent_shape_from_npz(current[LATENTS_NAME])
        if manifest_patch:
            manifest.update(manifest_patch)
        if touch:
            manifest["updated_at"] = _utc_now()
        data, written = _build_pack_bytes(manifest, current)
        _atomic_write_bytes(file_path, data)
        return written


# ----------------------------------------------------------------- public API


def slugify_character_name(name: str) -> str:
    token = _SLUG_RE.sub("-", str(name or "").strip())
    token = token.strip("-._") or "character"
    return token[:80]


def stage_create_still(
    data: bytes, *, suffix: str = ".png", dest_dir: Path | None = None
) -> Path:
    """Write a Create upload to a unique refs file so old sidecars cannot stick."""
    from .paths import refs_dir

    folder = dest_dir if dest_dir is not None else refs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    ext = str(suffix or ".png")
    if not ext.startswith("."):
        ext = f".{ext}"
    ext = ext.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        ext = ".png"
    dest = folder / f"character_create_{uuid4().hex}{ext}"
    dest.write_bytes(data)
    return dest


def unique_character_path(name: str, *, dest_dir: Path | None = None) -> Path:
    folder = dest_dir if dest_dir is not None else characters_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stem = slugify_character_name(name)
    candidate = folder / f"{stem}.vtm"
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        candidate = folder / f"{stem}-{i}.vtm"
        if not candidate.exists():
            return candidate
    raise CharacterPackError("Could not pick a unique character filename")


def _as_float16(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float16)


def write_character_pack(
    dest: Path | str,
    *,
    name: str,
    preview_rgb: np.ndarray,
    keypoints: np.ndarray,
    ref_latent: np.ndarray,
    ref_face_latent: np.ndarray | None,
    image_size: int,
    skip_crop: bool,
    source_name: str = "",
    fit: dict[str, Any] | None = None,
    source_bytes: bytes | None = None,
    source_suffix: str = ".png",
    pose_keys: list | None = None,
    blendshapes: dict | None = None,
    model: dict | None = None,
    meta: dict | None = None,
    created_at: str | None = None,
) -> Path:
    """Write a validated v2 ``.vtm`` zip atomically. Overwrites ``dest``.

    ``fit`` / ``pose_keys`` / ``blendshapes`` are written whenever they are not
    ``None`` (an explicit empty container is kept as an empty member).
    """
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.shape != (37, 4):
        raise CharacterPackError(f"Expected keypoints (37, 4), got {kps.shape}")
    preview = np.asarray(preview_rgb)
    if preview.ndim != 3 or preview.shape[2] != 3:
        raise CharacterPackError(f"Expected HxWx3 preview, got {preview.shape}")
    latent = _as_float16(ref_latent)
    if latent.ndim < 3:
        raise CharacterPackError("Character latent has the wrong shape")
    face = None if ref_face_latent is None else _as_float16(ref_face_latent)
    if fit is not None and not isinstance(fit, dict):
        raise CharacterPackError("Character fit must be a JSON object")
    if blendshapes is not None and not isinstance(blendshapes, dict):
        raise CharacterPackError("Character blendshapes must be a JSON object")
    if pose_keys is not None and not isinstance(pose_keys, list):
        raise CharacterPackError("Character pose keys must be a list")
    path = Path(dest)
    now = _utc_now()
    preview_png = _png_bytes(preview)
    latents_buf = io.BytesIO()
    payload: dict[str, np.ndarray] = {"ref_latent": latent}
    if face is not None:
        payload["ref_face_latent"] = face
    np.savez(latents_buf, **payload)
    kps_buf = io.BytesIO()
    np.save(kps_buf, kps)
    members: dict[str, bytes] = {
        PREVIEW_NAME: preview_png,
        THUMB_NAME: _thumb_from_png(preview_png),
        LATENTS_NAME: latents_buf.getvalue(),
        KEYPOINTS_NAME: kps_buf.getvalue(),
    }
    if fit is not None:
        members[FIT_NAME] = _json_bytes(fit)
    if pose_keys is not None:
        members[POSE_KEYS_NAME] = _json_bytes({"keys": list(pose_keys)})
    if blendshapes is not None:
        members[BLENDSHAPES_NAME] = _json_bytes(blendshapes)
    if source_bytes:
        members[f"{SOURCE_NAME_PREFIX}{_source_suffix(source_suffix)}"] = bytes(source_bytes)
    manifest = {
        "name": str(name or path.stem),
        "image_size": int(image_size),
        "skip_crop": bool(skip_crop),
        "has_face_latent": face is not None,
        "source_name": _file_name_only(source_name),
        "created_at": str(created_at or now),
        "updated_at": now,
        "model": _model_info(model, image_size=int(image_size), latent_shape=latent.shape),
        "meta": sanitize_pack_meta(meta),
    }
    data, _ = _build_pack_bytes(manifest, members)
    with _path_lock(path):
        _atomic_write_bytes(path, data)
    still = character_still_path(path.stem, dest_dir=path.parent)
    _atomic_write_bytes(still, preview_png)
    return path


def read_pack_fit(path: Path | str) -> dict[str, Any]:
    return read_pack_json(path, FIT_NAME, {})


def read_pack_json(path: Path | str, member: str, default: Any) -> Any:
    """One JSON member (hash-checked on v2); ``default`` if absent or not the same kind."""
    if not _is_safe_member_name(str(member)):
        raise CharacterPackError(f"Invalid character pack entry name: {member!r}")
    zf = _read_zip(Path(path))
    try:
        raw = _load_manifest(zf)
        return _parse_json(_read_member(zf, raw, str(member)), default)
    finally:
        zf.close()


def _json_member_bytes(member: str, data: Any) -> bytes:
    if member == POSE_KEYS_NAME:
        if isinstance(data, dict):
            data = {**data, "keys": _pose_keys_from_json(data)}
        else:
            data = {"keys": _pose_keys_from_json(data)}
    elif member in {FIT_NAME, BLENDSHAPES_NAME} and not isinstance(data, dict):
        raise CharacterPackError(f"{member} must be a JSON object")
    return _json_bytes(data)


def write_pack_json(path: Path | str, member: str, data: Any) -> None:
    """Replace one JSON member; keeps the rest, rehashes, upgrades v1 to v2."""
    name = str(member)
    if name == MANIFEST_NAME or not name.endswith(".json") or not _is_safe_member_name(name):
        raise CharacterPackError(f"Invalid character pack JSON entry: {member!r}")
    replace_pack_members(path, {name: _json_member_bytes(name, data)})


def replace_pack_members(path: Path | str, members: dict[str, bytes]) -> None:
    """Swap whole entries inside a ``.vtm``; every other entry is kept as-is."""
    _rewrite_pack(path, members={str(k): bytes(v) for k, v in members.items()})


def replace_pack_latents(
    path: Path | str,
    *,
    ref_latent: np.ndarray,
    ref_face_latent: np.ndarray | None,
    image_size: int,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Swap in latents re-encoded for another model and record that model.

    Everything else (keypoints, fit, keys, plan, source, meta) is kept.
    """
    latent = _as_float16(ref_latent)
    if latent.ndim < 3:
        raise CharacterPackError("Character latent has the wrong shape")
    face = None if ref_face_latent is None else _as_float16(ref_face_latent)
    buf = io.BytesIO()
    payload: dict[str, np.ndarray] = {"ref_latent": latent}
    if face is not None:
        payload["ref_face_latent"] = face
    np.savez(buf, **payload)
    size = int(image_size)
    return _rewrite_pack(
        path,
        members={LATENTS_NAME: buf.getvalue()},
        manifest_patch={
            "image_size": size,
            "has_face_latent": face is not None,
            "model": _model_info(model, image_size=size, latent_shape=latent.shape),
        },
    )


def write_pack_fit(path: Path | str, fit: dict[str, Any]) -> None:
    write_pack_json(path, FIT_NAME, fit)


def read_pack_pose_keys(path: Path | str) -> list[dict[str, Any]]:
    return _pose_keys_from_json(read_pack_json(path, POSE_KEYS_NAME, {}))


def write_pack_pose_keys(path: Path | str, keys: list[Any]) -> None:
    write_pack_json(path, POSE_KEYS_NAME, {"keys": list(keys or [])})


def read_pack_blendshapes(path: Path | str) -> dict[str, Any]:
    return read_pack_json(path, BLENDSHAPES_NAME, {})


def write_pack_blendshapes(path: Path | str, plan: dict[str, Any]) -> None:
    write_pack_json(path, BLENDSHAPES_NAME, plan)


def upgrade_character_pack(
    path: Path | str,
    *,
    fit: dict[str, Any] | None = None,
    pose_keys: list | None = None,
    blendshapes: dict | None = None,
    model: dict | None = None,
) -> bool:
    """Rewrite a pack as v2, folding in data that used to live in sidecars.

    Never overwrites pack data: ``fit`` only fills keys the pack's fit lacks,
    ``pose_keys`` / ``blendshapes`` only fill a missing or empty member, and
    ``model`` only fills unknown model fields. Returns True if the file changed.
    """
    file_path = Path(path)
    with _path_lock(file_path):
        raw, current = _load_for_rewrite(file_path)
        changed = _manifest_version(raw) < FORMAT_VERSION
        adds: dict[str, bytes] = {}
        if THUMB_NAME not in current and PREVIEW_NAME in current:
            adds[THUMB_NAME] = _thumb_from_png(current[PREVIEW_NAME])
        if isinstance(fit, dict) and fit:
            have = _parse_json(current.get(FIT_NAME), {})
            merged = {**fit, **have}
            if FIT_NAME not in current or merged != have:
                adds[FIT_NAME] = _json_bytes(merged)
        if pose_keys:
            have_keys = _pose_keys_from_json(_parse_json(current.get(POSE_KEYS_NAME), {}))
            if not have_keys:
                adds[POSE_KEYS_NAME] = _json_member_bytes(POSE_KEYS_NAME, list(pose_keys))
        if isinstance(blendshapes, dict) and blendshapes:
            if not _parse_json(current.get(BLENDSHAPES_NAME), {}):
                adds[BLENDSHAPES_NAME] = _json_bytes(blendshapes)
        manifest = _upgrade_manifest(
            raw, current, fallback_stamp=_file_stamp(file_path), model=model
        )
        if manifest.get("model") != raw.get("model"):
            changed = True
        if not adds and not changed:
            return False
        current.update(adds)
        manifest["updated_at"] = _utc_now()
        data, _ = _build_pack_bytes(manifest, current)
        _atomic_write_bytes(file_path, data)
        return True


def update_pack_meta(
    path: Path | str,
    *,
    author: str | None = None,
    license: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Change creator info in the manifest (``None`` = unchanged). Returns the manifest."""
    file_path = Path(path)
    with _path_lock(file_path):
        current = sanitize_pack_meta(peek_character_manifest(file_path).get("meta"))
        for key, value in (("author", author), ("license", license), ("description", description)):
            if value is not None:
                current[key] = value
        return _rewrite_pack(file_path, manifest_patch={"meta": sanitize_pack_meta(current)})


def read_character_preview_png(path: Path | str) -> bytes:
    zf = _read_zip(Path(path))
    try:
        raw = _load_manifest(zf)
        data = _read_member(zf, raw, PREVIEW_NAME)
    finally:
        zf.close()
    if data is None:
        raise CharacterPackError("Character pack is missing preview.png")
    return data


def read_character_thumb_png(path: Path | str) -> bytes:
    """Library thumbnail: ``thumb.png`` if packed, else made from the preview."""
    zf = _read_zip(Path(path))
    try:
        raw = _load_manifest(zf)
        thumb = _read_member(zf, raw, THUMB_NAME)
        preview = None if thumb is not None else _read_member(zf, raw, PREVIEW_NAME)
    finally:
        zf.close()
    if thumb is not None:
        return thumb
    if preview is None:
        raise CharacterPackError("Character pack is missing preview.png")
    return _thumb_from_png(preview)


def _pack_from_members(
    file_path: Path, raw: dict[str, Any], members: dict[str, bytes]
) -> CharacterPack:
    preview_png = members.get(PREVIEW_NAME)
    if preview_png is None:
        raise CharacterPackError("Character pack is missing preview.png")
    try:
        with Image.open(io.BytesIO(preview_png)) as img:
            preview = np.asarray(img.convert("RGB"))
    except Exception as exc:
        raise CharacterPackError("Character pack preview is unreadable") from exc
    try:
        with np.load(io.BytesIO(members[LATENTS_NAME])) as latents:
            ref_latent = np.asarray(latents["ref_latent"])
            face = None
            if bool(raw.get("has_face_latent")) and "ref_face_latent" in latents.files:
                face = np.asarray(latents["ref_face_latent"])
    except Exception as exc:
        raise CharacterPackError("Character pack latents are unreadable") from exc
    try:
        kps = np.asarray(
            np.load(io.BytesIO(members[KEYPOINTS_NAME]), allow_pickle=False),
            dtype=np.float32,
        )
    except Exception as exc:
        raise CharacterPackError("Character pack keypoints are unreadable") from exc
    if kps.shape != (37, 4):
        raise CharacterPackError(f"Expected keypoints (37, 4), got {kps.shape}")
    if ref_latent.ndim < 3:
        raise CharacterPackError("Character pack latent has the wrong shape")
    source_name = _source_member(members)
    image_size = int(raw.get("image_size") or 0)
    version = _manifest_version(raw)
    fallback = _file_stamp(file_path)
    created = str(raw.get("created_at") or fallback)
    model_raw = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    return CharacterPack(
        name=str(raw.get("name") or file_path.stem),
        preview_rgb=preview,
        keypoints=kps,
        ref_latent=ref_latent,
        ref_face_latent=face,
        image_size=image_size,
        skip_crop=bool(raw.get("skip_crop")),
        source_name=str(raw.get("source_name") or ""),
        has_face_latent=face is not None,
        fit=_parse_json(members.get(FIT_NAME), {}),
        version=version,
        pose_keys=_pose_keys_from_json(_parse_json(members.get(POSE_KEYS_NAME), {})),
        blendshapes=_parse_json(members.get(BLENDSHAPES_NAME), {}),
        model=_model_info(model_raw, image_size=image_size, latent_shape=ref_latent.shape),
        meta=sanitize_pack_meta(raw.get("meta")),
        created_at=created,
        updated_at=str(raw.get("updated_at") or created),
        source_bytes=None if source_name is None else members[source_name],
        source_suffix=_source_suffix(Path(source_name).suffix) if source_name else ".png",
    )


def read_character_pack(path: Path | str) -> CharacterPack:
    file_path = Path(path)
    zf = _read_zip(file_path)
    try:
        raw = _load_manifest(zf)
        members = _load_members(zf, raw, known_only=True)
    finally:
        zf.close()
    for required in (LATENTS_NAME, KEYPOINTS_NAME):
        if required not in members:
            what = "latents" if required == LATENTS_NAME else "keypoints"
            raise CharacterPackError(f"Character pack {what} are unreadable")
    return _pack_from_members(file_path, raw, members)


def validate_character_pack(path: Path | str) -> dict[str, Any]:
    """Full read + hash check (import). Returns the pack's v2 manifest."""
    pack = read_character_pack(path)
    manifest = pack.manifest()
    manifest["source_version"] = pack.version
    return manifest


def export_character_pack_bytes(path: Path | str) -> bytes:
    """Verified v2 bytes for download. A v1 file is upgraded in memory only."""
    file_path = Path(path)
    with _path_lock(file_path):
        pack = read_character_pack(file_path)
        raw, members = _load_for_rewrite(file_path)
        if pack.version >= FORMAT_VERSION and THUMB_NAME in members:
            return file_path.read_bytes()
    if THUMB_NAME not in members:
        members[THUMB_NAME] = _thumb_from_png(members[PREVIEW_NAME])
    manifest = _upgrade_manifest(raw, members, fallback_stamp=_file_stamp(file_path))
    data, _ = _build_pack_bytes(manifest, members)
    return data


def pack_model_matches(
    pack: CharacterPack,
    *,
    checkpoint: str,
    image_size: int,
    latent_shape: Iterable[int] | None,
) -> bool:
    """Can this pack's latents be used as-is by the loaded model?"""
    model = pack.model if isinstance(pack.model, dict) else {}
    try:
        pack_size = int(model.get("image_size") or pack.image_size or 0)
        want_size = int(image_size or 0)
    except (TypeError, ValueError):
        return False
    if not pack_size or pack_size != want_size:
        return False
    try:
        have_shape = tuple(int(v) for v in (model.get("latent_shape") or np.shape(pack.ref_latent)))
        want_shape = tuple(int(v) for v in (latent_shape or ()))
    except (TypeError, ValueError):
        return False
    if not have_shape or have_shape != want_shape:
        return False
    packed = _file_name_only(model.get("checkpoint")).lower()
    if not packed or packed == "unknown":
        return True
    return packed == _file_name_only(checkpoint).lower()


def character_still_path(ident: str, *, dest_dir: Path | None = None) -> Path:
    """Per-character folder: ``characters/<id>/preview.png``."""
    folder = dest_dir if dest_dir is not None else characters_dir()
    stem = Path(str(ident or "").strip()).name
    if not stem or stem in {".", ".."}:
        raise CharacterPackError("Invalid character id")
    return folder / stem / PREVIEW_NAME


def ensure_character_still(path: Path | str, *, dest_dir: Path | None = None) -> Path:
    """Write the pack preview into that character's folder if it is missing or stale."""
    pack = Path(path)
    library = dest_dir if dest_dir is not None else pack.parent
    still = character_still_path(pack.stem, dest_dir=library)
    if still.is_file() and still.stat().st_size > 0:
        try:
            if still.stat().st_mtime >= pack.stat().st_mtime:
                return still
        except OSError:
            return still
    _atomic_write_bytes(still, read_character_preview_png(pack))
    return still


def list_character_files(*, dest_dir: Path | None = None) -> list[Path]:
    folder = dest_dir if dest_dir is not None else characters_dir()
    if not folder.is_dir():
        return []
    found: list[Path] = []
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() == ".vtm":
            found.append(item)
        elif item.is_dir():
            found.extend(
                p for p in item.iterdir() if p.is_file() and p.suffix.lower() == ".vtm"
            )
    return sorted(found, key=lambda p: p.name.lower())


def character_card(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    raw: dict[str, Any] = {}
    try:
        raw = peek_character_manifest(file_path)
        name = str(raw.get("name") or file_path.stem)
    except CharacterPackError:
        name = file_path.stem
    meta = sanitize_pack_meta(raw.get("meta"))
    try:
        version = int(raw.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    ident = file_path.stem
    stamp = 0
    try:
        stamp = int(file_path.stat().st_mtime_ns)
    except OSError:
        pass
    return {
        "id": ident,
        "name": name,
        "path": display_path(file_path),
        "preview_url": f"/api/characters/{ident}/preview?v={stamp}",
        "thumb_url": f"/api/characters/{ident}/thumb?v={stamp}",
        "author": meta["author"],
        "version": version,
    }


def rename_character_pack(path: Path | str, name: str) -> dict[str, Any]:
    """Change the display name inside a pack. Filename / id stay the same."""
    file_path = Path(path)
    new_name = " ".join(str(name or "").split())
    if not new_name:
        raise CharacterPackError("Character name cannot be empty")
    _rewrite_pack(file_path, manifest_patch={"name": new_name[:80]})
    return character_card(file_path)


def delete_character_pack(path: Path | str, *, dest_dir: Path | None = None) -> None:
    """Remove a pack, pose keys, and leftover preview still so the library is empty."""
    file_path = Path(path)
    library = dest_dir if dest_dir is not None else characters_dir()
    stem = file_path.stem
    if file_path.is_file():
        file_path.unlink()
    for extra in (f"{stem}.keys.json", f"{stem}_pose_keys.json"):
        for sidecar in (file_path.with_name(extra), library / stem / extra):
            if sidecar.is_file():
                sidecar.unlink()
    still = character_still_path(stem, dest_dir=library)
    fit = still.with_name("fit.json")
    if fit.is_file():
        fit.unlink()
    if still.is_file():
        still.unlink()
    folder = still.parent
    try:
        if folder.resolve() == library.resolve():
            return
    except OSError:
        return
    if not folder.is_dir():
        return
    try:
        leftover = any(folder.iterdir())
    except OSError:
        return
    if not leftover:
        folder.rmdir()


def resolve_character_id(ident: str, *, dest_dir: Path | None = None) -> Path:
    folder = dest_dir if dest_dir is not None else characters_dir()
    stem = Path(str(ident).strip()).name
    if not stem or stem in {".", ".."}:
        raise CharacterPackError("Invalid character id")
    direct = folder / f"{stem}.vtm"
    if direct.is_file():
        return direct
    nested = folder / stem / f"{stem}.vtm"
    if nested.is_file():
        return nested
    if (folder / stem).is_dir():
        inside = [p for p in (folder / stem).iterdir() if p.suffix.lower() == ".vtm"]
        if len(inside) == 1:
            return inside[0]
    want = stem.lower()
    want_slug = slugify_character_name(stem).lower()
    for path in list_character_files(dest_dir=folder):
        if path.stem.lower() == want:
            return path
        try:
            name = str(peek_character_manifest(path).get("name") or "")
        except CharacterPackError:
            continue
        if name.lower() == want or slugify_character_name(name).lower() == want_slug:
            return path
    raise CharacterPackError(f"Character not found: {stem}")
