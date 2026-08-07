"""Runtime controller: StreamEngine + LivePoser + generate/stream workers."""

from __future__ import annotations

import base64
import io
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .developer import DEVELOPER
from .engine import (
    STREAM_DEFAULT_ID_CFG,
    STREAM_DEFAULT_POSE_CFG,
    STREAM_DEFAULT_STEPS,
    STREAM_TEMPORAL_EMA,
    StreamEngine,
    apply_live_deltas_to_ref,
    checkpoint_label,
    default_stream_checkpoint,
    list_stream_checkpoints,
    neutral_keypoints,
)
from .live_poser_client import (
    LivePoserTracker,
    body_method_kind,
    body_tracking_active,
    body_tracking_label,
    list_cameras,
    pick_default_camera_index,
)
from .live_retarget import is_mouth_closed_snap
from .paths import display_path, outputs_dir, resolve_user_path
from .pose_controller import draw_keypoint_mesh, nearest_keypoint, pixels_to_normalized

# Product defaults for options hidden when DEVELOPER is False.
# They must stay applied even though the UI cannot toggle them.
_PRODUCT_LOCKED_SETTINGS: dict[str, Any] = {
    "drive_pose": True,
    "use_iris": True,
    "use_body": True,
    "auto_sync_track": True,
}

Listener = Callable[[dict[str, Any]], None]


def _process_rss_bytes() -> int:
    """Current process working-set / RSS in bytes (best-effort)."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
        except Exception:
            pass
    return 0


def _estimate_model_load_bytes(ckpt: Path) -> int:
    """Expected RAM gain while loading DiT + SD-VAE (rough, for progress UI)."""
    ckpt_bytes = ckpt.stat().st_size if ckpt.is_file() else 400_000_000
    # Working set usually tracks checkpoint size plus ~1.5–2 GB for SD-VAE.
    return max(int(ckpt_bytes * 0.95 + 1.7e9), int(2.8e9))


def _image_to_jpeg_b64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class StreamRuntime:
    """Owns model, tracker, and background generate/stream loops for the API."""

    def __init__(self) -> None:
        default_ckpt = default_stream_checkpoint()
        self.engine = StreamEngine(
            checkpoint=default_ckpt,
            num_steps=STREAM_DEFAULT_STEPS,
            pose_cfg_scale=STREAM_DEFAULT_POSE_CFG,
            id_cfg_scale=STREAM_DEFAULT_ID_CFG,
            fast_mode=True,
        )
        self.tracker = LivePoserTracker()
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._status = {
            "state": "idle",
            # Lazy-load: importing torch/DiT/VAE on every UI open was a major RAM hit,
            # especially when orphan backends stacked. Model loads on first use.
            "message": "Idle — model loads on first generate / reference",
            "checkpoint": checkpoint_label(default_ckpt),
            "device": "",
            "error": "",
            "model_ready": False,
            "ref_ready": False,
            "streaming": False,
            "tracking": False,
            "busy": False,
            "fast_warming": False,
            "steps": STREAM_DEFAULT_STEPS,
            "track_fps": 2.0,
            "drive_pose": True,
            "show_mesh": True,
            "mirror": False,
            "use_iris": True,
            "use_body": True,
            "fast_mode": True,
            "batch2": False,
            "auto_sync_track": True,
            "gen_fps": 0.0,
            "timing": "",
            "reference_name": "",
            "reference_path": "",
            "camera_index": 0,
            "track_message": "Tracking off",
            "body_label": "",
            "progress": 0.0,
            "progress_label": "",
            "progress_kind": "",
            "compile_on": False,
            "compile_status": "off",
            "compile_detail": "",
        }
        if not DEVELOPER:
            self._status.update(_PRODUCT_LOCKED_SETTINGS)
        self._ref_path: Path | None = None
        self._last_image: Image.Image | None = None
        self._last_overlay_kps: np.ndarray | None = None
        self._driven_keypoints: np.ndarray | None = None
        self._last_good_keypoints: np.ndarray | None = None
        self._mesh_edited = False
        self._drag_kp_idx: int | None = None
        self._streaming = False
        self._frame_in_flight = False
        self._tracking = False
        # Live retarget state (webcam → character). Missing this made the mesh
        # draw raw webcam-scale keypoints and look tiny / "broken".
        self._live_origin_keypoints: np.ndarray | None = None
        self._live_origin_coord_space: str = "norm_crop"
        self._live_coord_space: str = "norm_crop"
        self._live_origin_crop_key: tuple[Any, ...] | None = None
        self._live_origin_skel_kind: str = "none"
        self._capture_session_seen: int = 0
        self._body_skel_method: str = "none"
        self._body_lost: bool = False
        self._mouth_snapped: bool = False
        self._last_live_preview_t: float = 0.0
        self._prev_stream_kps: np.ndarray | None = None
        self._ema_frame: np.ndarray | None = None
        self._fast_warmed: bool = False
        self._gen_queue: queue.Queue = queue.Queue(maxsize=1)
        self._model_load_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._gen_worker = threading.Thread(
            target=self._gen_worker_loop, name="rs-gen", daemon=True
        )
        self._gen_worker.start()
        self._track_stop = threading.Event()
        self._track_thread = threading.Thread(
            target=self._track_poll_loop, name="rs-track", daemon=True
        )
        self._track_thread.start()

    def add_listener(self, cb: Listener) -> None:
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb: Listener) -> None:
        with self._lock:
            if cb in self._listeners:
                self._listeners.remove(cb)

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception:
                pass

    def _set_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)
            self._status.update(self._compile_status_fields())
            snapshot = dict(self._status)
        self._emit({"type": "status", "status": snapshot})

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._status.update(self._compile_status_fields())
            return dict(self._status)

    def _compile_status_fields(self) -> dict[str, Any]:
        """Derive UI compile light from the engine (always fresh)."""
        try:
            st = str(getattr(self.engine, "compile_status", "off") or "off")
        except Exception:
            st = "off"
        try:
            detail = str(getattr(self.engine, "compile_detail", "") or "")
        except Exception:
            detail = ""
        return {
            "compile_status": st,
            "compile_on": st == "on",
            "compile_detail": detail,
        }

    def _set_progress(
        self,
        fraction: float,
        *,
        label: str,
        kind: str,
        message: str | None = None,
        **status_kwargs: Any,
    ) -> None:
        frac = max(0.0, min(1.0, float(fraction)))
        payload: dict[str, Any] = {
            "progress": frac,
            "progress_label": label,
            "progress_kind": kind,
            "busy": True,
        }
        payload.update(status_kwargs)
        if message is not None:
            payload["message"] = message
        else:
            payload["message"] = f"{label}  {frac * 100:.0f}%"
        self._set_status(**payload)

    def _clear_progress(self, **kwargs: Any) -> None:
        self._set_status(
            progress=0.0,
            progress_label="",
            progress_kind="",
            **kwargs,
        )

    def _run_with_ram_progress(
        self,
        fn: Callable[[], Any],
        *,
        label: str,
        kind: str,
        target_bytes: int,
    ) -> Any:
        """Poll process RAM while ``fn`` runs and map gain → progress 0..~0.92."""
        baseline = _process_rss_bytes()
        target = max(int(target_bytes), 1)
        stop = threading.Event()
        self._set_progress(0.03, label=label, kind=kind)

        def _poll() -> None:
            while not stop.wait(0.12):
                rss = _process_rss_bytes()
                if rss <= 0:
                    continue
                gained = max(0, rss - baseline)
                frac = min(0.92, gained / target)
                self._set_progress(max(0.03, frac), label=label, kind=kind)

        poller = threading.Thread(target=_poll, name="rs-ram-progress", daemon=True)
        poller.start()
        try:
            result = fn()
            self._set_progress(1.0, label=label, kind=kind, message=f"{label} done")
            return result
        finally:
            stop.set()
            poller.join(timeout=1.0)

    def _load_model(self, *, clear_busy: bool = True) -> None:
        try:
            ckpt = Path(self.engine.checkpoint)
            if not ckpt.is_file():
                from .engine import default_stream_checkpoint
                from .model_download import (
                    ensure_default_model,
                    has_local_checkpoints,
                    load_model_sources,
                )
                from .paths import models_dir as _models_dir

                dit = _models_dir()
                sources = load_model_sources()
                if sources.get("enabled") and not has_local_checkpoints():
                    self._set_progress(
                        0.02,
                        label="Downloading model",
                        kind="download",
                        message="Downloading VTM-ELF.pt into models/dit…",
                        busy=True,
                        state="downloading_models",
                        error="",
                    )
                    ensure_default_model(blocking=True)
                    ckpt = default_stream_checkpoint()
                    self.engine.checkpoint = ckpt

                if not ckpt.is_file():
                    msg = (
                        "No DiT models found. Place .pt files in "
                        f"{dit} or wait for the Hugging Face download to finish."
                    )
                    self._clear_progress(
                        state="need_models",
                        message=msg,
                        model_ready=False,
                        error=msg,
                        checkpoint="",
                        busy=False,
                    )
                    raise FileNotFoundError(msg)

            def _do_load() -> None:
                self.engine.load()

            self._run_with_ram_progress(
                _do_load,
                label="Loading model",
                kind="model",
                target_bytes=_estimate_model_load_bytes(ckpt),
            )
            device = str(getattr(self.engine, "device", ""))
            self._clear_progress(
                state="ready",
                message="Model ready",
                model_ready=True,
                device=device,
                error="",
                busy=not clear_busy,
                checkpoint=checkpoint_label(ckpt),
            )
        except Exception as exc:
            self._clear_progress(
                state="error",
                message="Model failed to load",
                error=str(exc),
                model_ready=False,
                busy=False,
            )
            raise

    def ensure_model(self, *, keep_busy: bool = False) -> None:
        """Load DiT+VAE once, on demand (not at UI open)."""
        with self._model_load_lock:
            if bool(self.status().get("model_ready")) and getattr(
                self.engine, "_ready", False
            ):
                return
            if getattr(self.engine, "_ready", False):
                self._set_status(
                    state="ready",
                    message="Model ready",
                    model_ready=True,
                    device=str(getattr(self.engine, "device", "")),
                )
                return
            self._load_model(clear_busy=not keep_busy)

    def list_checkpoints(self) -> list[dict[str, str]]:
        return [
            {"label": label, "path": display_path(path)}
            for label, path in list_stream_checkpoints()
        ]

    def set_checkpoint(self, path: str | Path) -> None:
        ckpt = resolve_user_path(path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        self._set_status(busy=True, message="Switching model…", error="")
        try:
            def _swap() -> None:
                self.engine.set_checkpoint(ckpt)

            # Re-encode path inside set_checkpoint may add more RAM; estimate fresh load.
            self._run_with_ram_progress(
                _swap,
                label="Loading model",
                kind="model",
                target_bytes=_estimate_model_load_bytes(ckpt),
            )
            self._clear_progress(
                busy=False,
                checkpoint=checkpoint_label(ckpt),
                message="Model ready",
                model_ready=True,
                state="ready",
                error="",
            )
        except Exception as exc:
            self._clear_progress(
                busy=False,
                error=str(exc),
                message="Model switch failed",
                model_ready=False,
            )
            raise

    def apply_reference(self, path: str | Path) -> dict[str, Any]:
        ref = resolve_user_path(path)
        if not ref.is_file():
            raise FileNotFoundError(f"Reference not found: {ref}")
        if self._streaming:
            raise RuntimeError("Stop the stream before applying a new reference")
        self._set_status(busy=True, message="Applying reference…", error="")
        try:
            # Load DiT+VAE first so status can say "Loading…" instead of looking stuck.
            if not getattr(self.engine, "_ready", False):
                self.ensure_model(keep_busy=True)

            def _on_ref_progress(frac: float, label: str) -> None:
                self._set_progress(frac, label=label, kind="reference")

            self._set_progress(0.05, label="Encoding reference…", kind="reference")
            self.engine.set_reference(ref, on_progress=_on_ref_progress)
            if getattr(self.engine, "_ready", False):
                self._set_status(
                    model_ready=True,
                    state="ready",
                    device=str(getattr(self.engine, "device", "")),
                )
            self._ref_path = ref
            kps = getattr(self.engine, "_ref_keypoints", None)
            if kps is not None:
                self._last_overlay_kps = np.asarray(kps, dtype=np.float32).copy()
                self._driven_keypoints = self._last_overlay_kps.copy()
                self._last_good_keypoints = self._last_overlay_kps.copy()
            self._mesh_edited = False
            self._reset_live_origin(reason="apply_reference")
            preview = None
            try:
                from .ref_pose_fit import reference_display_rgb

                arr = np.asarray(Image.open(ref).convert("RGB"))
                skip_crop = bool(getattr(self.engine, "_ref_skip_crop", False))
                preview = Image.fromarray(
                    reference_display_rgb(
                        arr,
                        skip_crop=skip_crop,
                        image_size=int(getattr(self.engine, "image_size", 768)),
                    )
                )
                self._last_image = preview
            except Exception:
                preview = Image.open(ref).convert("RGB")
                self._last_image = preview
            frame = self._frame_payload(preview, self._last_overlay_kps)
            self._clear_progress(
                busy=True,
                ref_ready=True,
                reference_name=ref.name,
                reference_path=display_path(ref),
                message="Reference applied",
                error="",
            )
            self._emit({"type": "frame", **frame})
            # Fast path: compile + warmup so the first stream frame isn't a freeze.
            # Blocks here until compile finishes (progress bar stays visible).
            self._run_fast_warmup_if_needed()
            snap = self.status()
            self._set_status(
                busy=False,
                message=str(snap.get("message") or "Reference applied"),
            )
            return frame
        except Exception as exc:
            # Always clear busy — otherwise the UI sticks on "Applying reference…".
            self._clear_progress(
                busy=False, error=str(exc), message="Reference failed"
            )
            raise

    def _run_fast_warmup_if_needed(self, *, force: bool = False) -> None:
        """Compile + prime Fast kernels with a visible progress bar.

        Blocks the caller until compile/warmup finishes. The UI progress bar
        keeps moving during the long torch.compile stage.
        """
        if not bool(self.status().get("fast_mode")):
            return
        if not getattr(self.engine, "_ready", False):
            return
        if getattr(self.engine, "_ref_latent", None) is None:
            return
        if self._streaming:
            return
        compile_st = str(getattr(self.engine, "compile_status", "off") or "off")
        if (
            not force
            and self._fast_warmed
            and compile_st in {"on", "fail", "skip", "off"}
        ):
            return

        stages = self.engine.plan_warmup_stages()
        total = max(1, len(stages))

        def _on_warm(done: float | int, total_n: int, label: str) -> None:
            frac = float(done) / float(max(total_n, 1))
            # Keep the bar moving while a long stage (compile) is in progress.
            if float(done) < float(total_n):
                frac = min(0.97, max(0.04, frac))
            self._set_progress(
                frac,
                label=str(label or "Compiling… please wait"),
                kind="warmup",
                message=str(label or "Compiling… please wait"),
                busy=True,
                fast_warming=True,
            )

        self._set_status(
            busy=True,
            fast_warming=True,
            message="Compiling… please wait",
            error="",
        )
        self._set_progress(
            0.04,
            label="Compiling… please wait",
            kind="warmup",
            message="Compiling… please wait",
            fast_warming=True,
        )
        try:
            self.engine.warmup(
                num_steps=int(self.status().get("steps") or STREAM_DEFAULT_STEPS),
                batch_size=int(getattr(self.engine, "stream_batch_size", 1) or 1),
                on_progress=_on_warm,
            )
            self._fast_warmed = True
            compile_on = bool(getattr(self.engine, "compile_status", "") == "on")
            msg = (
                "Ready — torch.compile on"
                if compile_on
                else f"Ready — compile {getattr(self.engine, 'compile_status', 'off')}"
            )
            self._clear_progress(
                busy=False,
                fast_warming=False,
                message=msg,
                error="",
            )
        except Exception as exc:
            self._fast_warmed = False
            self._clear_progress(
                busy=False,
                fast_warming=False,
                error=str(exc),
                message="Compile / warmup failed — continuing eager",
            )
            print(f"[warmup] failed: {exc}")

    def _ensure_compile_ready(self) -> None:
        """Block Generate / Stream until Fast compile+warmup finishes."""
        if not bool(self.status().get("fast_mode")):
            return
        if self.status().get("fast_warming"):
            raise RuntimeError("Compiling… please wait for the progress bar to finish")
        compile_st = str(getattr(self.engine, "compile_status", "off") or "off")
        if self._fast_warmed and compile_st in {"on", "fail", "skip", "off"}:
            return
        # Need a reference latent before we can compile/warmup.
        if getattr(self.engine, "_ref_latent", None) is None:
            return
        self._run_fast_warmup_if_needed(force=True)

    def update_settings(self, **kwargs: Any) -> dict[str, Any]:
        allowed = {
            "steps",
            "track_fps",
            "drive_pose",
            "show_mesh",
            "mirror",
            "use_iris",
            "use_body",
            "fast_mode",
            "batch2",
            "auto_sync_track",
            "camera_index",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not DEVELOPER:
            # Hidden product options stay locked to their defaults.
            for key, value in _PRODUCT_LOCKED_SETTINGS.items():
                if key in updates:
                    updates[key] = value
        if "steps" in updates:
            try:
                updates["steps"] = max(1, min(50, int(updates["steps"])))
            except (TypeError, ValueError):
                updates.pop("steps", None)
        with self._lock:
            self._status.update(updates)
            snap = dict(self._status)
        if "steps" in updates:
            try:
                self.engine.num_steps = int(updates["steps"])
            except Exception:
                pass
        if "mirror" in updates:
            self.tracker.mirror = bool(updates["mirror"])
            if self._tracking:
                self._reset_live_origin(reason="mirror")
                try:
                    self.tracker.reset_center()
                except Exception:
                    pass
        if "use_iris" in updates:
            self.tracker.use_iris = bool(updates["use_iris"])
        if "use_body" in updates:
            self.tracker.use_skeleton = bool(updates["use_body"])
        if "fast_mode" in updates:
            want = bool(updates["fast_mode"])
            try:
                self.engine.set_fast_mode(want)
                if not want:
                    self._fast_warmed = False
                if want and not self._streaming:
                    # Compile/warmup in-thread so the UI progress bar can show.
                    self._fast_warmed = False
                    self._run_fast_warmup_if_needed(force=True)
                    snap = self.status()
            except Exception as exc:
                self._set_status(error=str(exc), fast_mode=False)
                snap = self.status()
        if "batch2" in updates:
            try:
                self.engine.set_stream_batch_size(2 if updates["batch2"] else 1)
                # Batch shape change needs a fresh compile capture when Fast is on.
                if (
                    bool(self.status().get("fast_mode"))
                    and getattr(self.engine, "_model_compiled", False)
                    and not self._streaming
                ):
                    try:
                        self.engine._restore_eager_model()
                        self.engine._compile_failed = False
                    except Exception:
                        pass
                    self._fast_warmed = False
                    self._run_fast_warmup_if_needed(force=True)
                    snap = self.status()
            except Exception:
                pass
        self._emit({"type": "status", "status": snap})
        return snap

    def list_cameras(self) -> list[dict[str, Any]]:
        cams = list_cameras()
        return [{"index": i, "name": name} for i, name in cams]

    def preferred_camera(self) -> int:
        try:
            return int(pick_default_camera_index(list_cameras()))
        except Exception:
            return 0

    def _reset_live_origin(self, *, reason: str = "") -> None:
        self._live_origin_keypoints = None
        self._live_origin_crop_key = None
        self._live_origin_skel_kind = "none"
        try:
            self.engine._last_driven_body = None
        except Exception:
            pass
        if reason:
            print(f"[pose-diag] live origin reset ({reason})")

    @staticmethod
    def _snapshot_coord_key(snap: Any) -> tuple[Any, ...] | None:
        crop = getattr(snap, "crop", None)
        if crop is None:
            return None
        try:
            return (
                round(float(getattr(crop, "x0", 0.0)), 2),
                round(float(getattr(crop, "y0", 0.0)), 2),
                round(float(getattr(crop, "w", 0.0)), 2),
                round(float(getattr(crop, "h", 0.0)), 2),
                int(getattr(snap, "image_size", 0) or 0),
                bool(getattr(snap, "mirrored", False)),
            )
        except Exception:
            return None

    @staticmethod
    def _origin_ready(
        live: np.ndarray,
        *,
        method: str | None,
        lost: bool,
    ) -> bool:
        """Enough face (and body, when tracked) visibility to lock origin."""
        k = np.asarray(live, dtype=np.float32)
        if k.shape != (37, 4):
            return False
        face_vis = int(np.sum(k[:28, 3] >= 0.5))
        if face_vis < 18:
            return False
        kind = body_method_kind(method)
        if kind == "tracked" and not lost:
            body_vis = int(np.sum(k[30:37, 3] >= 0.5))
            return body_vis >= 4
        # Synthetic / no-body: face alone is enough.
        return True

    def start_tracking(self) -> None:
        if self._tracking:
            return
        if self._streaming:
            raise RuntimeError("Stop stream before starting tracking")
        st = self.status()
        cam = int(st.get("camera_index") or 0)
        self.tracker.mirror = bool(st.get("mirror"))
        use_iris = bool(st.get("use_iris", True))
        use_body = bool(st.get("use_body", True))
        if not DEVELOPER:
            use_iris = True
            use_body = True
        self.tracker.use_iris = use_iris
        self.tracker.use_skeleton = use_body
        self._reset_live_origin(reason="start_tracking")
        self._capture_session_seen = 0
        self._set_status(busy=True, message="Starting tracking…", track_message="Starting…", error="")
        try:
            self.tracker.start(cam, camera_name=f"Camera {cam}")
            self._tracking = True
            self._set_status(
                busy=False,
                tracking=True,
                message="Tracking on",
                track_message="Tracking on — auto-centering…",
            )
        except Exception as exc:
            self._tracking = False
            self._set_status(
                busy=False,
                tracking=False,
                error=str(exc),
                message="Tracking failed",
                track_message="Tracking failed",
            )
            raise

    def stop_tracking(self) -> None:
        if not self._tracking:
            return
        try:
            self.tracker.stop(settle=True)
        except Exception:
            pass
        self._tracking = False
        self._reset_live_origin(reason="stop_tracking")
        self._set_status(
            tracking=False,
            message="Tracking off",
            track_message="Tracking off",
            body_label="",
        )

    def calibrate_ref(self) -> None:
        if self._ref_path is None or not self._ref_path.is_file():
            raise RuntimeError("Pick a reference image first")
        self._set_status(busy=True, message="Calibrating reference…", error="")
        try:
            self.ensure_model()
            self._set_status(busy=True, message="Calibrating reference…")
            kps = self.engine.calibrate_reference(self._ref_path, flip_tta=True)
            self._last_overlay_kps = np.asarray(kps, dtype=np.float32).copy()
            self._driven_keypoints = self._last_overlay_kps.copy()
            self._last_good_keypoints = self._last_overlay_kps.copy()
            self._mesh_edited = False
            if self._last_image is not None:
                self._emit(
                    {
                        "type": "frame",
                        **self._frame_payload(self._last_image, self._last_overlay_kps),
                    }
                )
            self._set_status(
                busy=False,
                message="Reference calibrated",
                track_message="Calibrated",
                error="",
            )
        except Exception as exc:
            self._set_status(busy=False, error=str(exc), message="Calibrate failed")
            raise

    def recenter(self) -> None:
        if not self._tracking:
            raise RuntimeError("Start tracking first")
        self._reset_live_origin(reason="recenter")
        try:
            self.tracker.center()
        except Exception as exc:
            # Origin will re-lock from the next good frame.
            self._set_status(
                message=f"Recenter pending: {exc}",
                track_message="Waiting for face to re-center…",
            )
            return
        self._set_status(message="Recentered", track_message="Recentered — re-locking origin…")

    def _current_keypoints(self) -> np.ndarray | None:
        if self._mesh_edited and self._driven_keypoints is not None:
            return self._driven_keypoints
        st = self.status()
        drive = bool(st.get("drive_pose", True))
        if not DEVELOPER:
            drive = True
        if self._tracking and drive:
            driven = self._retarget_live_to_character()
            if driven is not None:
                return driven
        if self._driven_keypoints is not None:
            return self._driven_keypoints
        if self._last_good_keypoints is not None:
            return self._last_good_keypoints
        ref_kps = getattr(self.engine, "_ref_keypoints", None)
        if ref_kps is not None:
            return np.asarray(ref_kps, dtype=np.float32).copy()
        return neutral_keypoints()

    def _retarget_live_to_character(self) -> np.ndarray | None:
        """Map webcam keypoints onto the character ref (constrained retarget)."""
        snap = self.tracker.latest_snapshot()
        if snap is None or getattr(snap, "keypoints_norm", None) is None:
            return None

        capture_session = int(getattr(snap, "capture_session", 0) or 0)
        if capture_session and capture_session != self._capture_session_seen:
            if self._capture_session_seen:
                self._reset_live_origin(reason=f"capture_session {capture_session}")
                try:
                    self.tracker.reset_center()
                except Exception:
                    pass
            self._capture_session_seen = capture_session

        live = np.asarray(snap.keypoints_norm, dtype=np.float32)
        if live.shape != (37, 4):
            return None

        coord_space = str(getattr(snap, "coord_space", None) or "norm_crop")
        self._live_coord_space = coord_space
        method = (
            str(getattr(snap, "skeleton_method", None) or "")
            or (
                str(getattr(snap.bridge, "skel_method", ""))
                if getattr(snap, "bridge", None) is not None
                else ""
            )
            or "none"
        )
        lost = bool(getattr(snap, "body_lost", False))
        self._body_skel_method = method
        self._body_lost = lost

        crop_key = self._snapshot_coord_key(snap)
        if (
            self._live_origin_keypoints is not None
            and self._live_origin_crop_key is not None
            and crop_key is not None
            and crop_key != self._live_origin_crop_key
        ):
            self._reset_live_origin(reason="crop_changed")

        if self._live_origin_keypoints is None:
            if self._origin_ready(live, method=method, lost=lost):
                self._live_origin_keypoints = live.copy()
                self._live_origin_coord_space = coord_space
                self._live_origin_skel_kind = body_method_kind(method)
                self._live_origin_crop_key = crop_key
                try:
                    self.tracker.center()
                except Exception as exc:
                    print(f"[pose-diag] auto-center skipped: {exc}")
                else:
                    print("[pose-diag] auto-centered origin + head calib")
                self._set_status(track_message="Tracking on · origin locked")
            else:
                # Keep showing the character pose until origin locks.
                ref_kps = getattr(self.engine, "_ref_keypoints", None)
                if ref_kps is not None:
                    hold = np.asarray(ref_kps, dtype=np.float32).copy()
                    self._last_overlay_kps = hold
                    return hold
                return None

        ref_kps = getattr(self.engine, "_ref_keypoints", None)
        if ref_kps is None:
            return None

        rel = getattr(snap, "rel", None)
        head_yaw = float(rel.yaw) if rel is not None else None
        head_pitch = float(rel.pitch) if rel is not None else None
        head_roll = float(rel.roll) if rel is not None else None
        head_tx = float(rel.nx) if rel is not None and bool(rel.calibrated) else None
        head_ty = float(rel.ny) if rel is not None and bool(rel.calibrated) else None

        try:
            driven = apply_live_deltas_to_ref(
                ref_kps,
                live,
                self._live_origin_keypoints,
                body_method=method,
                body_lost=lost,
                prev_body=getattr(self.engine, "_last_driven_body", None),
                live_coord_space=coord_space,
                origin_coord_space=self._live_origin_coord_space,
                head_yaw_deg=head_yaw,
                head_pitch_deg=head_pitch,
                head_roll_deg=head_roll,
                head_tx_norm=head_tx,
                head_ty_norm=head_ty,
                # Live operator path: expression travel caps stay soft.
                limit_face=False,
                limit_brows=False,
                limit_eyes=False,
                limit_nose=False,
                limit_mouth=False,
                reference_rig=getattr(self.engine, "_ref_rig", None),
            )
        except Exception as exc:
            print(f"[pose-diag] retarget failed: {exc}")
            return self._driven_keypoints

        try:
            self.engine._last_driven_body = np.asarray(driven[30:37], dtype=np.float32).copy()
        except Exception:
            pass

        driven = np.asarray(driven, dtype=np.float32)
        self._driven_keypoints = driven.copy()
        self._last_good_keypoints = driven.copy()
        self._last_overlay_kps = driven.copy()
        try:
            self._mouth_snapped = bool(
                is_mouth_closed_snap(live, self._live_origin_keypoints)
            )
        except Exception:
            self._mouth_snapped = False
        return driven

    def _ensure_ref(self) -> None:
        if getattr(self.engine, "_ref_latent", None) is not None:
            return
        if self._ref_path and self._ref_path.is_file():
            self.apply_reference(self._ref_path)
            return
        raise RuntimeError("Pick a reference image first")

    def generate_once(self) -> None:
        if self._streaming or self.status().get("busy"):
            raise RuntimeError("Busy — stop the stream or wait")
        if self.status().get("fast_warming"):
            raise RuntimeError("Compiling… please wait for the progress bar to finish")
        self.ensure_model()
        self._ensure_ref()
        self._ensure_compile_ready()
        kps = self._current_keypoints()
        if kps is None:
            raise RuntimeError("No pose available yet — start tracking or apply a reference")
        steps = int(self.status().get("steps") or STREAM_DEFAULT_STEPS)
        self._set_status(busy=True, message="Generating…")
        self._enqueue_generate(steps=steps, streaming=False, save=True, keypoints=kps)

    def start_stream(self) -> None:
        if self._streaming:
            return
        if self.status().get("fast_warming"):
            raise RuntimeError("Compiling… please wait for the progress bar to finish")
        self.ensure_model()
        self._ensure_ref()
        self._ensure_compile_ready()
        self._streaming = True
        self._frame_in_flight = False
        self._prev_stream_kps = None
        self._ema_frame = None
        self._set_status(streaming=True, message="Streaming", busy=False)
        self._schedule_next_frame()

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        self._streaming = False
        while True:
            try:
                job = self._gen_queue.get_nowait()
            except queue.Empty:
                break
            else:
                try:
                    self._gen_queue.task_done()
                except Exception:
                    pass
                if job is None:
                    try:
                        self._gen_queue.put_nowait(None)
                    except queue.Full:
                        pass
                    break
        self._frame_in_flight = False
        self._prev_stream_kps = None
        self._ema_frame = None
        self._set_status(streaming=False, busy=False, message="Stream stopped")

    def _schedule_next_frame(self) -> None:
        if not self._streaming or self._frame_in_flight:
            return
        self._frame_in_flight = True
        kps = self._current_keypoints()
        if kps is None:
            self._frame_in_flight = False
            threading.Timer(0.05, self._schedule_next_frame).start()
            return
        current = np.asarray(kps, dtype=np.float32)
        batch_n = max(1, int(getattr(self.engine, "stream_batch_size", 1) or 1))
        if batch_n > 1 and self._prev_stream_kps is not None:
            # Older pose first, then newest — displayed in that order after one DiT call.
            keypoints = np.stack([self._prev_stream_kps, current], axis=0)
        else:
            keypoints = current
        self._prev_stream_kps = current.copy()
        steps = int(self.status().get("steps") or STREAM_DEFAULT_STEPS)
        self._enqueue_generate(steps=steps, streaming=True, save=False, keypoints=keypoints)

    def _enqueue_generate(
        self,
        *,
        steps: int,
        streaming: bool,
        save: bool,
        keypoints: np.ndarray,
    ) -> None:
        job = {
            "steps": steps,
            "streaming": streaming,
            "save": save,
            "keypoints": np.asarray(keypoints, dtype=np.float32),
            "sanitize": "constrained",
        }
        try:
            self._gen_queue.put_nowait(job)
        except queue.Full:
            try:
                _ = self._gen_queue.get_nowait()
                self._gen_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._gen_queue.put_nowait(job)
            except queue.Full:
                pass

    def _blend_display_frame(self, image: Image.Image) -> Image.Image:
        """Light temporal EMA so Batch×2 A/B samples don't hard-pop each other."""
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        alpha = float(STREAM_TEMPORAL_EMA)
        alpha = max(0.05, min(1.0, alpha))
        if self._ema_frame is None or self._ema_frame.shape != arr.shape:
            self._ema_frame = arr
            return image
        self._ema_frame = alpha * arr + (1.0 - alpha) * self._ema_frame
        out = np.clip(self._ema_frame, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def _gen_worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                job = self._gen_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                steps = int(job["steps"])
                keypoints = job["keypoints"]
                streaming = bool(job.get("streaming"))
                images, elapsed = self.engine.generate_batch_from_keypoints(
                    keypoints,
                    num_steps=steps,
                    sanitize=job.get("sanitize", "constrained"),
                )
                timings = dict(getattr(self.engine, "last_timings", {}) or {})
                used_batch = getattr(self.engine, "last_target_keypoints_batch", None)
                if used_batch is None:
                    used_batch = np.asarray(keypoints, dtype=np.float32)
                    if used_batch.ndim == 2:
                        used_batch = used_batch[None, ...]

                if job.get("save") and images:
                    out = outputs_dir() / "last.png"
                    images[-1].save(out)

                n = max(1, len(images))
                per = float(elapsed) / float(n) if elapsed > 0 else 0.0
                for i, image in enumerate(images):
                    if streaming:
                        image = self._blend_display_frame(image)
                    kps_i = (
                        used_batch[i]
                        if used_batch is not None and i < used_batch.shape[0]
                        else None
                    )
                    is_last = i == len(images) - 1
                    self._on_frame(
                        image,
                        per,
                        timings,
                        kps_i,
                        streaming=streaming,
                        schedule_next=is_last if streaming else False,
                    )
            except Exception as exc:
                streaming = bool(job.get("streaming"))
                if streaming:
                    self._streaming = False
                    self._frame_in_flight = False
                    self._set_status(
                        streaming=False,
                        busy=False,
                        error=str(exc),
                        message="Stream error",
                    )
                else:
                    self._set_status(busy=False, error=str(exc), message="Generate failed")
            finally:
                try:
                    self._gen_queue.task_done()
                except Exception:
                    pass

    def _on_frame(
        self,
        image: Image.Image,
        elapsed: float,
        timings: dict,
        keypoints: np.ndarray | None,
        *,
        streaming: bool,
        schedule_next: bool | None = None,
    ) -> None:
        self._last_image = image
        if keypoints is not None:
            self._last_overlay_kps = np.asarray(keypoints, dtype=np.float32)
        fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        with self._lock:
            prev = float(self._status.get("gen_fps") or 0.0)
            ema = fps if prev <= 0 else (0.7 * prev + 0.3 * fps)
            den = float(timings.get("denoise_s", timings.get("denoise", 0)) or 0)
            dec = float(timings.get("decode_s", timings.get("decode", 0)) or 0)
            timing = f"denoise {den:.2f}s | decode {dec:.2f}s" if timings else ""
            batch_n = int(float(timings.get("batch", 0) or 0))
            if batch_n > 1:
                timing = f"{timing} | batch×{batch_n}" if timing else f"batch×{batch_n}"
            self._status.update(
                gen_fps=ema,
                timing=timing,
                busy=False if not streaming else self._status.get("busy", False),
                message="Streaming" if streaming else "Generated",
            )
        frame = self._frame_payload(image, self._last_overlay_kps)
        frame["elapsed"] = elapsed
        frame["fps"] = ema if streaming else fps
        frame["timing"] = timing
        self._emit({"type": "frame", **frame})
        self._emit({"type": "status", "status": self.status()})
        if streaming:
            # Mid-batch frames keep in-flight; only the last arms the next DiT call.
            if schedule_next is False:
                return
            self._frame_in_flight = False
            if self._streaming:
                self._schedule_next_frame()

    def _frame_payload(
        self, image: Image.Image | None, keypoints: np.ndarray | None
    ) -> dict[str, Any]:
        st = self.status()
        show_mesh = bool(st.get("show_mesh"))
        display = image
        if display is not None and show_mesh and keypoints is not None:
            try:
                display = draw_keypoint_mesh(
                    display.copy(),
                    keypoints,
                    skeleton_lost=bool(self._body_lost),
                    mouth_snapped=bool(self._mouth_snapped),
                )
            except Exception:
                display = image
        payload: dict[str, Any] = {
            "image": _image_to_jpeg_b64(display) if display is not None else None,
            "width": int(display.width) if display is not None else 0,
            "height": int(display.height) if display is not None else 0,
            "keypoints": None,
        }
        if keypoints is not None:
            payload["keypoints"] = np.asarray(keypoints, dtype=np.float32).tolist()
        return payload

    def mesh_press(self, x: float, y: float) -> None:
        if not self.status().get("show_mesh") or self._last_image is None:
            return
        kps = self._last_overlay_kps
        if kps is None:
            return
        w, h = self._last_image.size
        idx = nearest_keypoint(kps, x, y, w, h)
        self._drag_kp_idx = idx

    def mesh_drag(self, x: float, y: float) -> None:
        if self._drag_kp_idx is None or self._last_image is None:
            return
        kps = (
            np.asarray(self._last_overlay_kps, dtype=np.float32).copy()
            if self._last_overlay_kps is not None
            else None
        )
        if kps is None:
            return
        w, h = self._last_image.size
        nx, ny = pixels_to_normalized(x, y, w, h)
        kps[self._drag_kp_idx, 0] = nx
        kps[self._drag_kp_idx, 1] = ny
        kps[self._drag_kp_idx, 3] = 1.0
        self._last_overlay_kps = kps
        self._driven_keypoints = kps.copy()
        self._mesh_edited = True
        if self._last_image is not None:
            self._emit({"type": "frame", **self._frame_payload(self._last_image, kps)})

    def mesh_release(self) -> None:
        self._drag_kp_idx = None

    def mesh_reset(self) -> None:
        self._mesh_edited = False
        self._drag_kp_idx = None
        ref_kps = getattr(self.engine, "_ref_keypoints", None)
        if ref_kps is not None:
            self._last_overlay_kps = np.asarray(ref_kps, dtype=np.float32).copy()
            self._driven_keypoints = self._last_overlay_kps.copy()
            if self._last_image is not None:
                self._emit(
                    {
                        "type": "frame",
                        **self._frame_payload(self._last_image, self._last_overlay_kps),
                    }
                )

    def _track_poll_loop(self) -> None:
        while not self._track_stop.is_set():
            time.sleep(0.05)
            if not self._tracking:
                continue
            try:
                snap = self.tracker.latest_snapshot()
                method = self._body_skel_method
                lost = self._body_lost
                if snap is not None:
                    method = (
                        str(getattr(snap, "skeleton_method", None) or "")
                        or method
                        or "none"
                    )
                    lost = bool(getattr(snap, "body_lost", False))
                    self._body_skel_method = method
                    self._body_lost = lost
                    # Keep driven overlay fresh while idle (not only on generate).
                    if not self._streaming and not self._mesh_edited:
                        now = time.time()
                        if now - self._last_live_preview_t >= 0.12:
                            self._last_live_preview_t = now
                            driven = self._retarget_live_to_character()
                            if driven is not None and self._last_image is not None:
                                self._emit(
                                    {
                                        "type": "frame",
                                        **self._frame_payload(self._last_image, driven),
                                    }
                                )
                kind = body_method_kind(method)
                body_on = body_tracking_active(method, lost=lost)
                label = (
                    body_tracking_label(method, lost=lost)
                    if body_on or kind not in {"none", ""}
                    else ""
                )
                msg = "Tracking on"
                if self._live_origin_keypoints is None:
                    msg = "Tracking on · auto-centering…"
                age = self.tracker.age_seconds()
                if age < 1e8 and self._live_origin_keypoints is not None:
                    msg = f"Tracking on · {age:.1f}s"
                if self.tracker.error:
                    msg = f"Tracking error: {self.tracker.error}"
                self._set_status(track_message=msg, body_label=label or "")
            except Exception:
                pass

    def shutdown(self) -> None:
        self.stop_stream()
        try:
            self.stop_tracking()
        except Exception:
            pass
        self._worker_stop.set()
        self._track_stop.set()
        try:
            self._gen_queue.put_nowait(None)
        except Exception:
            pass
        # Drop CUDA / model refs so RAM can return after close.
        try:
            eng = self.engine
            eng.model = None
            eng.vae = None
            if hasattr(eng, "_ready"):
                eng._ready = False
        except Exception:
            pass
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


_runtime: StreamRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> StreamRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = StreamRuntime()
        return _runtime


def shutdown_runtime() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.shutdown()
            _runtime = None
