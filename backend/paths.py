"""Resolve project / packaged data paths (self-contained install root only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def package_root() -> Path:
    """Directory that contains `backend/`, `ui/`, `vendor/`, `models/`, and `characters/`.

    Defaults to the folder above `backend/`; `VTM_NOBLE_ROOT` overrides it.
    """
    env_root = (
        os.environ.get("VTM_NOBLE_ROOT", "").strip()
        or os.environ.get("REAL_STREAM_ROOT", "").strip()
    )
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            return p.resolve()

    # backend/paths.py -> backend/ -> install root
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Install / GitHub root (same as package_root — no parent monorepo)."""
    return package_root()


def _path_roots() -> list[Path]:
    """Candidate roots for relative display / resolve (package only)."""
    return [package_root().resolve()]


def display_path(path: Path | str) -> str:
    """Package-relative posix path for UI / API responses."""
    p = Path(path)
    try:
        resolved = p.resolve() if p.is_absolute() else (package_root() / p).resolve()
    except OSError:
        return Path(path).as_posix().replace("\\", "/")
    root = package_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix().replace("\\", "/")


def _posix_rel(path: Path | str) -> str:
    return Path(path).as_posix().replace("\\", "/")


def _legacy_character_rel(path: Path | str) -> Path | None:
    """Map old ``models/characters/...`` stores onto ``characters/...``."""
    raw = Path(str(path))
    posix = _posix_rel(raw)
    prefix = "models/characters/"
    if posix.startswith(prefix):
        rest = posix[len(prefix) :]
        return Path("characters") / rest if rest else Path("characters")
    if posix == "models/characters":
        return Path("characters")
    root = package_root().resolve()
    try:
        abs_path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        rel = abs_path.relative_to(root)
    except (OSError, ValueError):
        return None
    rel_posix = rel.as_posix()
    if rel_posix.startswith(prefix):
        rest = rel_posix[len(prefix) :]
        return Path("characters") / rest if rest else Path("characters")
    if rel_posix == "models/characters":
        return Path("characters")
    return None


def resolve_user_path(path: Path | str) -> Path:
    """Resolve an absolute or package-relative path from the UI/API."""
    p = Path(path)
    candidate = p if p.is_absolute() else (package_root() / p).resolve()
    if candidate.exists():
        return candidate
    remapped = _legacy_character_rel(p)
    if remapped is not None:
        return (package_root() / remapped).resolve()
    return candidate


def ensure_under_models(path: Path | str) -> Path:
    """Resolve *path* and require it stay under the package ``models/`` tree."""
    resolved = resolve_user_path(path).resolve()
    root = models_root().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint must be inside models/ (got {display_path(resolved)})"
        ) from exc
    return resolved


def data_dir() -> Path:
    """App JSON / refs / lock — kept under ``models/``."""
    return models_root()


def models_root() -> Path:
    """User-facing models tree (DiT download + bundled trackers)."""
    root = package_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def torch_compile_cache_dir() -> Path:
    """Persistent torch.compile / Triton kernels (survives app restarts)."""
    d = models_root() / "cache" / "torch_inductor"
    d.mkdir(parents=True, exist_ok=True)
    return d


def triton_cache_dir() -> Path:
    d = models_root() / "cache" / "triton"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configure_torch_compile_cache() -> None:
    """Point inductor / Triton at models/cache before ``import torch`` when possible."""
    inductor = str(torch_compile_cache_dir())
    triton = str(triton_cache_dir())
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", inductor)
    os.environ.setdefault("TRITON_CACHE_DIR", triton)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")


def models_dir() -> Path:
    """DiT checkpoint directory (relative models/dit; downloaded on setup)."""
    preferred = models_root() / "dit"
    preferred.mkdir(parents=True, exist_ok=True)

    # Migrate typo folder models/iamge → models/dit (one-time, best-effort).
    typo = models_root() / "iamge"
    if typo.is_dir():
        try:
            for src in typo.iterdir():
                if not src.is_file() or src.name.startswith("."):
                    continue
                dest = preferred / src.name
                if not dest.exists():
                    src.replace(dest)
            if not any(typo.iterdir()):
                typo.rmdir()
        except OSError:
            pass

    return preferred


def refs_dir() -> Path:
    d = data_dir() / "refs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _migrate_legacy_characters(dest: Path) -> None:
    """Move leftover packs from ``models/characters`` into ``characters/``."""
    legacy = package_root() / "models" / "characters"
    if not legacy.is_dir():
        return
    try:
        if legacy.resolve() == dest.resolve():
            return
        entries = list(legacy.iterdir())
    except OSError:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for src in entries:
        if src.name == ".gitkeep":
            continue
        target = dest / src.name
        if target.exists():
            continue
        try:
            src.replace(target)
        except OSError:
            pass
    leftover = []
    try:
        leftover = [p for p in legacy.iterdir() if p.name != ".gitkeep"]
    except OSError:
        return
    if leftover:
        return
    try:
        gitkeep = legacy / ".gitkeep"
        if gitkeep.is_file():
            gitkeep.unlink()
        legacy.rmdir()
    except OSError:
        pass


def characters_dir() -> Path:
    """User character packs (``.vtm``) — package-relative ``characters/``."""
    d = package_root() / "characters"
    d.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_characters(d)
    return d


def blendshapes_dir() -> Path:
    """Track Lab plan shapes copied for each character (``.json`` sidecars)."""
    d = models_root() / "blendshapes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trackers_dir() -> Path:
    """Small tracker weights shipped with the app / downloaded on setup."""
    preferred = models_root() / "trackers"
    preferred.mkdir(parents=True, exist_ok=True)
    if any(preferred.iterdir()):
        return preferred
    legacy = package_root() / "data" / "trackers"
    if legacy.is_dir() and any(legacy.iterdir()):
        return legacy
    return preferred


def outputs_dir() -> Path:
    """Legacy folder — do not write app frames or logs here."""
    return package_root() / "outputs"


def ui_dist_dir() -> Path:
    """Return the built Vite output."""
    return package_root() / "ui" / "dist"


def _first_existing(*candidates: Path, marker: str | None = None) -> Path:
    for path in candidates:
        if marker is None:
            if path.is_dir():
                return path
        elif (path / marker).exists():
            return path
    return candidates[0]


def torch_train_dir() -> Path:
    """DiT inference helpers (`inference_keypoint`, `models`, `vae`, …)."""
    return _first_existing(
        package_root() / "vendor" / "torch_train",
        marker="inference_keypoint.py",
    )


def tools_dir() -> Path:
    """Vendored tools (live-poser + OpenSeeFace + pose-traker)."""
    return _first_existing(
        package_root() / "vendor" / "tools",
        marker="live-poser",
    )


def live_poser_dir() -> Path:
    return tools_dir() / "live-poser"


def pose_traker_dir() -> Path:
    return tools_dir() / "pose-traker"


def anime_face_detector_src() -> Path:
    return pose_traker_dir() / "anime-face-detector" / "src"


def openseeface_dir() -> Path:
    return _first_existing(
        tools_dir() / "openseeface",
        marker="tracker.py",
    )


def default_ref_candidates() -> list[Path]:
    """Reference stills under models/refs only (user uploads / shipped defaults)."""
    return [
        refs_dir() / "default.png",
        refs_dir() / "train_char_1.png",
        refs_dir() / "upload.png",
    ]


def default_ref_path() -> Path:
    for path in default_ref_candidates():
        if path.is_file():
            return path
    return default_ref_candidates()[0]


def ensure_import_paths() -> None:
    """Add vendored tool roots to sys.path for tracker / DiT helpers."""
    for path in (
        torch_train_dir(),
        live_poser_dir(),
        pose_traker_dir(),
        anime_face_detector_src(),
        openseeface_dir(),
    ):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
