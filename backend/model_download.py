"""Download DiT checkpoints into models/dit (config-driven)."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .paths import display_path, models_dir, models_root, package_root

_lock = threading.RLock()
_state: dict[str, Any] = {
    "status": "idle",  # idle | downloading | done | error | unavailable
    "message": "",
    "progress": 0.0,
    "current_file": "",
    "error": "",
    "files_done": 0,
    "files_total": 0,
    "bytes_done": 0,
    "bytes_total": 0,
}

# Minimum size to treat an existing file as a real checkpoint (not a stub).
_MIN_CKPT_BYTES = 1_000_000
NEW_MODEL_DAYS = 30
_CATALOG_TTL_S = 10 * 60
_catalog_lock = threading.Lock()
_catalog_cache: tuple[float, list[dict[str, Any]]] | None = None

# When True, print ASCII progress to stderr (Smart Build / CLI). Off inside the live app UI.
_console_progress = False
_progress_line_len = 0
_progress_last_newline_at = 0.0
# Cap hard so IDE terminals (RawUI often reports 120 while the panel is narrower)
# cannot wrap every \r update into a scroll waterfall.
_PROGRESS_MAX_COLS = 76


def _use_console_progress() -> bool:
    return bool(_console_progress) or (
        stream_runtime_is_idle() and (sys.stderr.isatty() or sys.stdout.isatty())
    )


def _progress_supports_inplace() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def stream_runtime_is_idle() -> bool:
    try:
        from . import stream as stream_mod

        return stream_mod._runtime is None
    except Exception:
        return True


def enable_console_progress(enabled: bool = True) -> None:
    global _console_progress
    _console_progress = bool(enabled)


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    if n >= 1024**3:
        return f"{n / (1024**3):.2f} GB"
    if n >= 1024**2:
        return f"{n / (1024**2):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{int(n)} B"


def _print_bar(
    *,
    label: str,
    frac: float,
    detail: str = "",
    width: int = 28,
    force: bool = False,
) -> None:
    global _progress_line_len, _progress_last_newline_at

    frac = max(0.0, min(1.0, float(frac)))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * frac
    line = f"    [{bar}] {pct:5.1f}%  {label}"
    if detail:
        line = f"{line}  {detail}"
    if len(line) >= _PROGRESS_MAX_COLS:
        line = line[: _PROGRESS_MAX_COLS - 1]

    if not _progress_supports_inplace():
        # Redirected stderr: \r does not overwrite — throttle to avoid spam.
        now = time.monotonic()
        if (
            not force
            and (now - _progress_last_newline_at) < 1.5
            and frac < 0.995
        ):
            return
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        _progress_last_newline_at = now
        _progress_line_len = 0
        return

    pad = max(0, _progress_line_len - len(line))
    sys.stderr.write("\r" + line + (" " * pad))
    sys.stderr.flush()
    _progress_line_len = len(line)


def _finish_bar(label: str = "done", detail: str = "") -> None:
    global _progress_line_len
    _print_bar(label=label, frac=1.0, detail=detail, force=True)
    if _progress_supports_inplace():
        sys.stderr.write("\n")
        sys.stderr.flush()
    _progress_line_len = 0


def _copy_with_progress(src: Path, dest: Path, *, label: str) -> None:
    total = src.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    done = 0
    chunk = 1024 * 1024
    show = _use_console_progress()
    if show:
        print(f"  copying {label} ({_fmt_bytes(total)})...", flush=True)
    with src.open("rb") as inp, tmp.open("wb") as out:
        while True:
            buf = inp.read(chunk)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if show and total > 0:
                _print_bar(
                    label=label,
                    frac=done / total,
                    detail=f"{_fmt_bytes(done)} / {_fmt_bytes(total)}",
                )
                _set_state(
                    progress=min(0.99, done / max(total, 1)),
                    bytes_done=done,
                    bytes_total=total,
                    message=f"Copying {label}…",
                )
    tmp.replace(dest)
    if show:
        _finish_bar(label=label, detail=_fmt_bytes(total))


def model_sources_path() -> Path:
    preferred = models_root() / "model_sources.json"
    if preferred.is_file():
        return preferred
    legacy = package_root() / "data" / "model_sources.json"
    if legacy.is_file():
        return legacy
    return preferred


def load_model_sources() -> dict[str, Any]:
    path = model_sources_path()
    if not path.is_file():
        return {
            "enabled": False,
            "message": "",
            "hf_repo": "",
            "hf_revision": "main",
            "files": [],
            "http_files": [],
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "enabled": False,
            "message": f"Invalid model_sources.json: {exc}",
            "hf_repo": "",
            "hf_revision": "main",
            "files": [],
            "http_files": [],
        }


def list_local_checkpoints() -> list[dict[str, Any]]:
    dit = models_dir()
    if not dit.is_dir():
        return []
    return [
        {"name": p.name, "path": display_path(p), "bytes": p.stat().st_size}
        for p in sorted(dit.iterdir())
        if p.is_file()
        and not p.name.startswith(".")
        and p.stat().st_size >= _MIN_CKPT_BYTES
    ]


def has_local_checkpoints() -> bool:
    return bool(list_local_checkpoints())


def models_status() -> dict[str, Any]:
    dit = models_dir()
    checkpoints = list_local_checkpoints()
    sources = load_model_sources()
    with _lock:
        download = dict(_state)
    from .model_checklist import scan_models

    checklist = scan_models()
    return {
        "dit_dir": display_path(dit),
        "has_models": bool(checkpoints),
        "checkpoints": checkpoints,
        "download_configured": bool(sources.get("enabled")),
        "download_message": str(sources.get("message") or ""),
        "download": download,
        "checklist": checklist,
        "setup_ok": bool(checklist.get("success")),
    }


def download_status() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def _set_state(**kwargs: Any) -> None:
    with _lock:
        _state.update(kwargs)
    _notify_runtime()


def _notify_runtime() -> None:
    """Push download progress into the live app status (best-effort)."""
    try:
        from . import stream as stream_mod

        # Don't instantiate StreamRuntime just for CLI / build downloads.
        if stream_mod._runtime is None:
            return

        st = download_status()
        status = str(st.get("status") or "idle")
        rt = stream_mod._runtime
        if status == "downloading":
            rt._set_progress(
                float(st.get("progress") or 0.0),
                label=str(st.get("current_file") or "Downloading model"),
                kind="download",
                message=str(st.get("message") or "Downloading model…"),
                busy=True,
                state="downloading_models",
                error="",
            )
        elif status == "done":
            rt._clear_progress(
                busy=False,
                state="idle",
                message=str(st.get("message") or "Model download complete"),
                error="",
            )
        elif status == "error":
            rt._clear_progress(
                busy=False,
                state="need_models",
                message="Model download failed",
                error=str(st.get("error") or "Download failed"),
            )
    except Exception:
        pass


def start_model_download(names: list[str] | None = None) -> dict[str, Any]:
    sources = load_model_sources()
    if not sources.get("enabled"):
        _set_state(
            status="unavailable",
            message=str(sources.get("message") or "Download not configured"),
            error="",
            progress=0.0,
        )
        return download_status()

    with _lock:
        if _state.get("status") == "downloading":
            return dict(_state)

    thread = threading.Thread(
        target=_run_download,
        args=(sources, names),
        name="vtm-model-download",
        daemon=True,
    )
    _set_state(
        status="downloading",
        message="Starting download…",
        progress=0.0,
        current_file="",
        error="",
        files_done=0,
        files_total=0,
    )
    thread.start()
    return download_status()


def _asset_jobs(sources: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize model_sources.json into download jobs with dest paths."""
    jobs: list[dict[str, Any]] = []
    assets = list(sources.get("assets") or [])
    if assets:
        for item in assets:
            if not isinstance(item, dict):
                continue
            hf = str(item.get("hf") or item.get("filename") or "").strip()
            dest = str(item.get("dest") or "").strip()
            if not hf or not dest:
                continue
            jobs.append(
                {
                    "kind": "hf",
                    "hf": hf,
                    "dest": dest,
                    "min_bytes": int(item.get("min_bytes") or _MIN_CKPT_BYTES),
                }
            )
    else:
        # Legacy: files[] land in models/dit/
        for name in sources.get("files") or []:
            name = str(name).strip()
            if not name:
                continue
            jobs.append(
                {
                    "kind": "hf",
                    "hf": name,
                    "dest": f"models/dit/{Path(name).name}",
                    "min_bytes": _MIN_CKPT_BYTES,
                }
            )

    for item in sources.get("http_files") or []:
        if isinstance(item, dict) and item.get("url") and item.get("name"):
            dest = str(item.get("dest") or f"models/dit/{item['name']}").strip()
            jobs.append(
                {
                    "kind": "http",
                    "url": str(item["url"]),
                    "dest": dest,
                    "min_bytes": int(item.get("min_bytes") or _MIN_CKPT_BYTES),
                }
            )
    return jobs


def _dest_path(rel: str) -> Path:
    return (package_root() / rel).resolve()


def _asset_present(job: dict[str, Any]) -> bool:
    path = _dest_path(str(job["dest"]))
    return path.is_file() and path.stat().st_size >= int(job.get("min_bytes") or 1)


def missing_asset_jobs(sources: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    sources = sources or load_model_sources()
    return [j for j in _asset_jobs(sources) if not _asset_present(j)]


def _is_dit_weight_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".pt") or lower.endswith(".pth") or lower.endswith(".ckpt")


def hub_checkpoint_names() -> set[str]:
    """Filenames that came from Hugging Face (catalog), not custom local drops."""
    from .ui_session import load_ui_session

    names: set[str] = set()
    for raw in load_ui_session().get("hub_files") or []:
        name = Path(str(raw)).name
        if _is_dit_weight_name(name):
            names.add(name)
    sources = load_model_sources()
    for job in _asset_jobs(sources):
        dest = str(job.get("dest") or "")
        if dest.replace("\\", "/").startswith("models/dit/"):
            name = Path(dest).name
            if _is_dit_weight_name(name):
                names.add(name)
    for raw in sources.get("files") or []:
        name = Path(str(raw)).name
        if _is_dit_weight_name(name):
            names.add(name)
    return names


def _local_dit_names() -> set[str]:
    names: set[str] = set()
    dit = models_dir()
    if not dit.is_dir():
        return names
    try:
        entries = list(dit.iterdir())
    except OSError:
        return names
    nested: list[Path] = []
    for path in entries:
        if path.is_dir() and not path.name.startswith("."):
            nested.append(path)
            continue
        if path.is_file() and _is_dit_weight_name(path.name):
            names.add(path.name)
    for child in nested:
        try:
            kids = list(child.iterdir())
        except OSError:
            continue
        for path in kids:
            if path.is_file() and _is_dit_weight_name(path.name):
                names.add(path.name)
    return names


def _hub_dit_listing(sources: dict[str, Any]) -> list[dict[str, Any]]:
    """Hub DiT filenames plus last-commit time when Hugging Face provides it."""
    repo = str(sources.get("hf_repo") or "").strip()
    if not repo:
        return []
    revision = str(sources.get("hf_revision") or "main")
    try:
        from huggingface_hub import HfApi

        tree = list(HfApi().list_repo_tree(repo, revision=revision, recursive=True))
    except Exception as exc:
        print(f"Hub catalog ping skipped ({exc})")
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in tree:
        rel = str(getattr(entry, "path", "") or "").replace("\\", "/").strip()
        if not rel:
            continue
        parts = Path(rel).parts
        name = Path(rel).name
        if not _is_dit_weight_name(name):
            continue
        if len(parts) > 2:
            continue
        if len(parts) == 2 and parts[0] not in {"dit", "models"}:
            continue
        if name in seen:
            continue
        seen.add(name)
        published = None
        last = getattr(entry, "last_commit", None)
        date = getattr(last, "date", None) if last is not None else None
        if date is not None:
            try:
                published = date if getattr(date, "tzinfo", None) else date.replace(tzinfo=timezone.utc)
            except Exception:
                published = None
        rows.append({"name": name, "hf": rel, "dest": f"models/dit/{name}", "published": published})
    return rows


def hub_catalog_offers(
    *,
    now: datetime | None = None,
    force: bool = False,
    listing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Missing hub DiT weights. ``is_new`` is true for 30 days after publish."""
    sources = load_model_sources()
    if not sources.get("enabled"):
        return []
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    global _catalog_cache
    if listing is None:
        with _catalog_lock:
            cached = _catalog_cache
            if not force and cached and (time.time() - cached[0]) < _CATALOG_TTL_S:
                listing = cached[1]
            else:
                listing = _hub_dit_listing(sources)
                _catalog_cache = (time.time(), listing)
    owned = _local_dit_names()
    cutoff = stamp - timedelta(days=NEW_MODEL_DAYS)
    offers: list[dict[str, Any]] = []
    for row in listing or []:
        name = str(row.get("name") or "")
        if not name or name in owned:
            continue
        published = row.get("published")
        is_new = bool(published and published >= cutoff)
        offers.append(
            {
                "name": name,
                "label": Path(name).stem,
                "path": str(row.get("dest") or f"models/dit/{name}"),
                "is_new": is_new,
                "published": published.isoformat() if hasattr(published, "isoformat") else "",
                "badge": "New" if is_new else "Available",
            }
        )
    offers.sort(key=lambda item: (not item["is_new"], str(item["label"]).lower()))
    return offers


def _hf_dit_jobs(sources: dict[str, Any]) -> list[dict[str, Any]]:
    """Ping the Hub repo for new DiT weights; never overwrite files already on disk."""
    repo = str(sources.get("hf_repo") or "").strip()
    if not repo:
        return []
    revision = str(sources.get("hf_revision") or "main")
    names: list[str] = []
    jobs: list[dict[str, Any]] = []
    seen_dest: set[str] = set()
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(repo, revision=revision)
    except Exception as exc:
        print(f"Hub catalog ping skipped ({exc})")
        files = []

    for rel in files:
        rel_s = str(rel).replace("\\", "/").strip()
        if not rel_s or rel_s.endswith("/"):
            continue
        parts = Path(rel_s).parts
        name = Path(rel_s).name
        if not _is_dit_weight_name(name):
            continue
        if len(parts) > 2:
            continue
        if len(parts) == 2 and parts[0] not in {"dit", "models"}:
            continue
        names.append(name)
        dest = f"models/dit/{name}"
        if dest in seen_dest:
            continue
        seen_dest.add(dest)
        jobs.append(
            {
                "kind": "hf",
                "hf": rel_s,
                "dest": dest,
                "min_bytes": _MIN_CKPT_BYTES,
            }
        )

    if names:
        from .ui_session import save_ui_session

        save_ui_session(hub_files=sorted(set(names)))
    return jobs


def _merged_download_jobs(sources: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in list(_asset_jobs(sources)) + _hf_dit_jobs(sources):
        dest = str(job.get("dest") or "")
        if not dest or dest in seen:
            continue
        seen.add(dest)
        jobs.append(job)
    return jobs


def ensure_default_model(*, blocking: bool = False) -> dict[str, Any]:
    """Ensure configured HF assets exist locally; download any that are missing.

    Existing files are never deleted or overwritten. Hub is pinged for *new*
    DiT checkpoints only.
    """
    sources = load_model_sources()
    jobs = _merged_download_jobs(sources) if sources.get("enabled") else []
    missing = [j for j in jobs if not _asset_present(j)]

    if not missing:
        # Prefer checklist pass when available.
        try:
            from .model_checklist import checklist_success

            ok = checklist_success()
        except Exception:
            ok = has_local_checkpoints()
        with _lock:
            if _state.get("status") not in {"done", "downloading"}:
                _set_state(
                    status="done" if ok else "idle",
                    message=(
                        "All configured models present"
                        if ok
                        else f"Models present in {display_path(models_dir())}"
                    ),
                    progress=1.0,
                    error="",
                    current_file="",
                )
        return download_status()

    if not sources.get("enabled"):
        _set_state(
            status="unavailable",
            message=str(sources.get("message") or "Download not configured"),
            error="",
            progress=0.0,
        )
        return download_status()

    with _lock:
        already_downloading = _state.get("status") == "downloading"

    if already_downloading:
        return wait_for_download(timeout=3600.0) if blocking else download_status()

    if blocking:
        _set_state(
            status="downloading",
            message="Starting download…",
            progress=0.0,
            current_file="",
            error="",
            files_done=0,
            files_total=0,
        )
        _run_download(sources)
        return download_status()

    return start_model_download()


def wait_for_download(*, timeout: float = 3600.0) -> dict[str, Any]:
    """Block until a background download finishes (or times out)."""
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        st = download_status()
        if st.get("status") in {"done", "error", "unavailable", "idle"}:
            if st.get("status") == "idle" and has_local_checkpoints():
                return st
            if st.get("status") != "idle":
                return st
        if has_local_checkpoints() and st.get("status") != "downloading":
            return st
        time.sleep(0.5)
    _set_state(
        status="error",
        message="Download timed out",
        error=f"Timed out after {timeout:.0f}s",
    )
    return download_status()


def _run_download(sources: dict[str, Any], names: list[str] | None = None) -> None:
    try:
        hf_repo = str(sources.get("hf_repo") or "").strip()
        jobs = _merged_download_jobs(sources)
        if names:
            want = {Path(str(n)).name for n in names if str(n).strip()}
            jobs = [job for job in jobs if Path(str(job.get("dest") or "")).name in want]
        if not jobs:
            raise RuntimeError("model_sources.json has no files to download")
        if any(j["kind"] == "hf" for j in jobs) and not hf_repo:
            raise RuntimeError("model_sources.json missing hf_repo")

        _set_state(files_total=len(jobs), files_done=0)
        for i, job in enumerate(jobs):
            dest = _dest_path(str(job["dest"]))
            label = Path(str(job["dest"])).name
            min_bytes = int(job.get("min_bytes") or 1)
            if dest.is_file() and dest.stat().st_size >= min_bytes:
                _set_state(
                    current_file=label,
                    message=f"Already present: {job['dest']}",
                    files_done=i + 1,
                    progress=(i + 1) / max(len(jobs), 1),
                )
                continue

            _set_state(
                current_file=label,
                message=f"Downloading {job['dest']}…",
                progress=i / max(len(jobs), 1),
            )
            if job["kind"] == "hf":
                _download_hf(
                    repo_id=hf_repo,
                    filename=str(job["hf"]),
                    revision=str(sources.get("hf_revision") or "main"),
                    dest_path=dest,
                    min_bytes=min_bytes,
                )
            else:
                _download_http(str(job["url"]), dest)
                if not dest.is_file() or dest.stat().st_size < min_bytes:
                    raise RuntimeError(f"HTTP download invalid at {dest}")
            _set_state(files_done=i + 1, progress=(i + 1) / max(len(jobs), 1))

        _set_state(
            status="done",
            message=f"Downloaded {len(jobs)} asset(s) from {hf_repo or 'HTTP'}",
            progress=1.0,
            current_file="",
            error="",
        )
        reload_runtime_after_download()
    except Exception as exc:
        _set_state(
            status="error",
            message="Download failed",
            error=str(exc),
        )


def _download_hf(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    dest_path: Path,
    min_bytes: int = _MIN_CKPT_BYTES,
) -> None:
    import shutil
    import tempfile

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for HF downloads") from exc

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    show = _use_console_progress()
    if show:
        print(f"  downloading {filename} from {repo_id}...", flush=True)
        print("  (Hugging Face progress below; large files can take several minutes)", flush=True)

    with tempfile.TemporaryDirectory(prefix="vtm-hf-") as tmp:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=tmp,
        )
        src = Path(path)
        if not src.is_file():
            raise RuntimeError(f"HF download missing file for {filename}")
        # Atomic-ish replace with progress (copy can look frozen on big checkpoints).
        label = Path(filename).name
        if show and src.stat().st_size >= 8 * 1024 * 1024:
            _copy_with_progress(src, dest_path, label=label)
        else:
            tmp_out = dest_path.with_suffix(dest_path.suffix + ".partial")
            shutil.copy2(src, tmp_out)
            tmp_out.replace(dest_path)

    if not dest_path.is_file() or dest_path.stat().st_size < min_bytes:
        raise RuntimeError(f"Download did not produce a valid file at {dest_path}")
    if show:
        print(f"  OK {dest_path.name} ({_fmt_bytes(dest_path.stat().st_size)})", flush=True)


def _download_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    show = _use_console_progress()
    label = dest.name
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            total = -1
            try:
                total = int(resp.headers.get("Content-Length") or -1)
            except Exception:
                total = -1
            if show:
                size_hint = _fmt_bytes(total) if total > 0 else "unknown size"
                print(f"  downloading {label} ({size_hint})...", flush=True)
            done = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if show:
                    if total > 0:
                        _print_bar(
                            label=label,
                            frac=done / total,
                            detail=f"{_fmt_bytes(done)} / {_fmt_bytes(total)}",
                        )
                        _set_state(
                            progress=min(0.99, done / total),
                            bytes_done=done,
                            bytes_total=total,
                            message=f"Downloading {label}…",
                        )
                    else:
                        _print_bar(
                            label=label,
                            frac=0.0,
                            detail=f"{_fmt_bytes(done)} downloaded",
                        )
        tmp.replace(dest)
        if show:
            _finish_bar(label=label, detail=_fmt_bytes(dest.stat().st_size))
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"HTTP download failed: {exc}") from exc
    finally:
        if tmp.exists() and not dest.exists():
            tmp.unlink(missing_ok=True)


def reload_runtime_after_download() -> None:
    """Best-effort: point StreamRuntime at the newly downloaded default ckpt."""
    try:
        from . import stream as stream_mod
        from .engine import checkpoint_label, default_stream_checkpoint

        if stream_mod._runtime is None:
            return
        ckpt = default_stream_checkpoint()
        if not ckpt.is_file():
            return
        rt = stream_mod._runtime
        # Don't force a full GPU load here — just update the selected path/label.
        with rt._lock:
            rt.engine.checkpoint = ckpt
        rt._set_status(
            checkpoint=checkpoint_label(ckpt),
            message=f"Checkpoint ready: {ckpt.name}",
        )
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m backend.model_download [--checklist-only] [--dry-run]

    --dry-run  download into a temp folder, verify, then delete (no lasting files)
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    checklist_only = "--checklist-only" in args
    dry_run = "--dry-run" in args
    enable_console_progress(True)

    if checklist_only:
        from .model_checklist import print_checklist

        return print_checklist()

    if dry_run:
        return _dry_run_download()

    print(f"DiT directory: {models_dir()}")
    print("Existing files are kept. Hub ping downloads only new missing checkpoints.")
    st = ensure_default_model(blocking=True)
    status = st.get("status")
    if status == "done" or has_local_checkpoints():
        print(st.get("message") or "OK")
        for ck in list_local_checkpoints():
            print(f"  - {ck['name']} ({ck['bytes']} bytes)")
        from .model_checklist import format_checklist_text, scan_models

        report = scan_models()
        print()
        print(format_checklist_text(report))
        return 0 if report.get("success") else 1
    print(st.get("message") or "Download failed", file=sys.stderr)
    if st.get("error"):
        print(st["error"], file=sys.stderr)
    return 1


def _dry_run_download() -> int:
    """Download configured HF assets into a temp tree, verify, then delete."""
    import shutil
    import tempfile

    sources = load_model_sources()
    if not sources.get("enabled"):
        print("Download not configured (model_sources.json enabled=false)", file=sys.stderr)
        return 1
    hf_repo = str(sources.get("hf_repo") or "").strip()
    jobs = [j for j in _asset_jobs(sources) if j["kind"] == "hf"]
    if not hf_repo or not jobs:
        print("No HF assets configured", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="vtm-dry-dl-"))
    print(f"Dry-run download -> temp {tmp}")
    print(f"  repo={hf_repo} assets={len(jobs)}")
    print("  Progress bars below mean the download is still running (not frozen).")
    try:
        enable_console_progress(True)
        for job in jobs:
            dest = tmp / str(job["dest"])
            print(f"  downloading {job['hf']} -> {job['dest']}...")
            _download_hf(
                repo_id=hf_repo,
                filename=str(job["hf"]),
                revision=str(sources.get("hf_revision") or "main"),
                dest_path=dest,
                min_bytes=int(job.get("min_bytes") or 1),
            )
            print(f"  OK {dest.name} ({dest.stat().st_size} bytes)")
        print("Dry-run succeeded - deleting temp download...")
        return 0
    except Exception as exc:
        print(f"Dry-run FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"Temp removed: {tmp}")


if __name__ == "__main__":
    raise SystemExit(main())
