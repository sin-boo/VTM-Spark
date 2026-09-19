"""Portable VTM character packs (``.vtm`` = zip, no compile artifacts).

A pack is the cooked identity another NVIDIA box can load without VAE-encoding
the still again: preview PNG + float16 latents + KEYPOINT_SCHEMA rest pose.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from .paths import characters_dir, display_path

FORMAT_ID = "vtm-character"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PREVIEW_NAME = "preview.png"
LATENTS_NAME = "latents.npz"
KEYPOINTS_NAME = "keypoints.npy"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


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

    def manifest(self) -> dict[str, Any]:
        return {
            "format": FORMAT_ID,
            "version": FORMAT_VERSION,
            "name": self.name,
            "image_size": int(self.image_size),
            "skip_crop": bool(self.skip_crop),
            "has_face_latent": bool(self.ref_face_latent is not None),
            "source_name": self.source_name,
        }


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
) -> Path:
    """Write a validated ``.vtm`` zip. Overwrites ``dest``."""
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.shape != (37, 4):
        raise CharacterPackError(f"Expected keypoints (37, 4), got {kps.shape}")
    preview = np.asarray(preview_rgb)
    if preview.ndim != 3 or preview.shape[2] != 3:
        raise CharacterPackError(f"Expected HxWx3 preview, got {preview.shape}")
    latent = _as_float16(ref_latent)
    face = None if ref_face_latent is None else _as_float16(ref_face_latent)
    pack = CharacterPack(
        name=str(name or Path(dest).stem),
        preview_rgb=preview,
        keypoints=kps,
        ref_latent=latent,
        ref_face_latent=face,
        image_size=int(image_size),
        skip_crop=bool(skip_crop),
        source_name=str(source_name or ""),
        has_face_latent=face is not None,
    )
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    preview_buf = io.BytesIO()
    Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8)).save(
        preview_buf, format="PNG"
    )
    latents_buf = io.BytesIO()
    payload: dict[str, np.ndarray] = {"ref_latent": latent}
    if face is not None:
        payload["ref_face_latent"] = face
    np.savez(latents_buf, **payload)
    kps_buf = io.BytesIO()
    np.save(kps_buf, kps)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(pack.manifest(), indent=2))
        zf.writestr(PREVIEW_NAME, preview_buf.getvalue())
        zf.writestr(LATENTS_NAME, latents_buf.getvalue())
        zf.writestr(KEYPOINTS_NAME, kps_buf.getvalue())
    path.write_bytes(buf.getvalue())
    still = character_still_path(path.stem, dest_dir=path.parent)
    still.parent.mkdir(parents=True, exist_ok=True)
    still.write_bytes(preview_buf.getvalue())
    return path


def _read_zip(path: Path) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise CharacterPackError(f"Not a character pack: {path.name}") from exc
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
    if version != FORMAT_VERSION:
        raise CharacterPackError(f"Unsupported character pack version {version}")
    return raw


def read_character_preview_png(path: Path | str) -> bytes:
    zf = _read_zip(Path(path))
    try:
        _load_manifest(zf)
        return zf.read(PREVIEW_NAME)
    except KeyError as exc:
        raise CharacterPackError("Character pack is missing preview.png") from exc
    finally:
        zf.close()


def read_character_pack(path: Path | str) -> CharacterPack:
    file_path = Path(path)
    zf = _read_zip(file_path)
    try:
        raw = _load_manifest(zf)
        try:
            preview = np.asarray(Image.open(io.BytesIO(zf.read(PREVIEW_NAME))).convert("RGB"))
        except Exception as exc:
            raise CharacterPackError("Character pack preview is unreadable") from exc
        try:
            latents = np.load(io.BytesIO(zf.read(LATENTS_NAME)))
            ref_latent = np.asarray(latents["ref_latent"])
            face = None
            if bool(raw.get("has_face_latent")) and "ref_face_latent" in latents.files:
                face = np.asarray(latents["ref_face_latent"])
        except Exception as exc:
            raise CharacterPackError("Character pack latents are unreadable") from exc
        try:
            kps = np.asarray(np.load(io.BytesIO(zf.read(KEYPOINTS_NAME))), dtype=np.float32)
        except Exception as exc:
            raise CharacterPackError("Character pack keypoints are unreadable") from exc
        if kps.shape != (37, 4):
            raise CharacterPackError(f"Expected keypoints (37, 4), got {kps.shape}")
        if ref_latent.ndim < 3:
            raise CharacterPackError("Character pack latent has the wrong shape")
        return CharacterPack(
            name=str(raw.get("name") or file_path.stem),
            preview_rgb=preview,
            keypoints=kps,
            ref_latent=ref_latent,
            ref_face_latent=face,
            image_size=int(raw.get("image_size") or 0),
            skip_crop=bool(raw.get("skip_crop")),
            source_name=str(raw.get("source_name") or ""),
            has_face_latent=face is not None,
        )
    finally:
        zf.close()


def validate_character_pack(path: Path | str) -> dict[str, Any]:
    pack = read_character_pack(path)
    return pack.manifest()


def character_still_path(ident: str, *, dest_dir: Path | None = None) -> Path:
    """Per-character folder: ``models/characters/<id>/preview.png``."""
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
    still.parent.mkdir(parents=True, exist_ok=True)
    still.write_bytes(read_character_preview_png(pack))
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
    try:
        raw = peek_character_manifest(file_path)
        name = str(raw.get("name") or file_path.stem)
    except CharacterPackError:
        name = file_path.stem
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
    }


def rename_character_pack(path: Path | str, name: str) -> dict[str, Any]:
    """Change the display name inside a pack. Filename / id stay the same."""
    file_path = Path(path)
    new_name = " ".join(str(name or "").split())
    if not new_name:
        raise CharacterPackError("Character name cannot be empty")
    new_name = new_name[:80]
    zf = _read_zip(file_path)
    try:
        raw = dict(_load_manifest(zf))
        raw["name"] = new_name
        others = [(item, zf.read(item)) for item in zf.namelist() if item != MANIFEST_NAME]
    finally:
        zf.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as dest:
        dest.writestr(MANIFEST_NAME, json.dumps(raw, indent=2))
        for item, data in others:
            dest.writestr(item, data)
    file_path.write_bytes(buf.getvalue())
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
