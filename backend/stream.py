"""Runtime controller: StreamEngine + LivePoser + generate/stream workers."""

from __future__ import annotations

import base64
import io
import queue
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from . import debug_log
from .developer import DEVELOPER
from .desk_boot import (
    new_boot_state,
    patch_boot_stage,
    previous_load_target,
    snapshot_boot,
)
from .pose_keys import (
    apply_keys,
    load_keys,
    params_from_controls,
    save_keys,
)
from .engine import (
    STREAM_COMPILE_MODEL,
    STREAM_DEFAULT_ID_CFG,
    STREAM_DEFAULT_POSE_CFG,
    STREAM_DEFAULT_STEPS,
    STREAM_HOLD_LAST,
    STREAM_BATCH_MAX,
    STREAM_INBETWEENS,
    STREAM_INTERPOLATE,
    STREAM_MAX_GEN_FPS,
    STREAM_TEMPORAL_EMA,
    StreamEngine,
    apply_live_deltas_to_ref,
    apply_overlay_drag_to_rest,
    discard_sidecar_keypoints,
    checkpoint_label,
    default_stream_checkpoint,
    is_stream_checkpoint_file,
    list_stream_checkpoints,
    remember_checkpoint_location,
    neutral_keypoints,
    _clip_batch,
    _clip_blend,
    _clip_cfg,
    _clip_inbetweens,
    _clip_max_fps,
    effective_inbetweens,
    face_pose_delta,
    gen_cap,
    gen_hold_s,
    interpolate_on,
    snap_alpha,
)
from .load_timing import StageClock, StageMeter
from .frame_interp import (
    PLAYOUT_QUEUE_MAX,
    SHOW_FPS_MAX,
    inbetween_maker,
    inbetween_pacing,
    inbetween_ts,
    playout_gap,
    lerp_stream_hair,
    lerp_stream_pose,
)
from .live_poser_client import (
    LivePoserTracker,
    body_method_kind,
    body_tracking_active,
    body_tracking_label,
    list_cameras,
    pick_default_camera_index,
)
from .hair_follow import (
    HairRig,
    build_hair_rig,
    follow_hair,
    rest_hair,
)
from .tracking.hair import create_hair_tracker, resolve_hair_weights
from .paths import characters_dir, data_dir, display_path, package_root, resolve_user_path


def _remembered_camera() -> tuple[int | None, str]:
    try:
        from track_lab.backend.cameras import load_camera_choice

        return load_camera_choice()
    except Exception:
        return None, ""
from .pose_controller import (
    NUM_KEYPOINTS,
    draw_hair_overlay,
    draw_keypoint_mesh,
    nearest_keypoint,
    overlay_visible_slots,
    pixels_to_normalized,
)
from .travel_box import (
    default_travel_box,
    apply_travel_box,
    changed_preview_axis,
    motion_caps,
    draw_travel_box,
    normalize_travel_box,
    preview_travel_pose,
)
from utils.hair import empty_hair_maps, rasterize_hair_maps

# Product defaults for options hidden when DEVELOPER is False.
# They must stay applied even though the UI cannot toggle them.
_PRODUCT_LOCKED_SETTINGS: dict[str, Any] = {
    "drive_pose": True,
    "use_iris": True,
    "use_body": True,
    "auto_sync_track": True,
    "fast_mode": True,
}

OVERLAY_PART_KEYS = (
    "show_outline",
    "show_brows",
    "show_eyes",
    "show_nose",
    "show_mouth",
    "show_iris_overlay",
    "show_skeleton",
)

# After Fast warmup, enable Batch×2 when VRAM is still under this fraction.
GPU_BATCH2_UTIL_LIMIT = 0.90


def pack_stream_batch(
    current: np.ndarray,
    current_hair: np.ndarray,
    prev: np.ndarray | None,
    prev_hair: np.ndarray | None,
    batch_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Poses for one DiT call: ``batch_n`` even steps from the last call's pose to now.

    Batch×2 used to stack [prev, current], which re-drew the pose already on
    screen and popped. Even steps (1/2, now; 1/3, 2/3, now; …) continue the
    motion with unseen poses only.

    The first call of a stream has no previous pose; it still sends ``batch_n``
    copies of now. torch.compile and the decoder were warmed at this batch
    size, and a lone batch-1 call first recompiled both — ~14 s with no frame.
    """
    now = np.asarray(current, dtype=np.float32)
    hair = np.asarray(current_hair, dtype=np.float32)
    n = int(batch_n)
    if n <= 1:
        return now, hair
    if prev is None:
        return np.stack([now] * n, axis=0), np.stack([hair] * n, axis=0)
    poses = [lerp_stream_pose(prev, now, (i + 1) / n) for i in range(n - 1)] + [now]
    hairs = [lerp_stream_hair(prev_hair, hair, (i + 1) / n) for i in range(n - 1)] + [hair]
    return np.stack(poses, axis=0), np.stack(hairs, axis=0)

Listener = Callable[[dict[str, Any]], None]

# Start stream's one bar: moving weights back to the GPU, then warmup, then
# "Starting stream…" until the first picture.
GPU_MOVE_SHARE = 0.10
# FPS readout window: pictures shown in the last second.
SHOWN_FPS_WINDOW_S = 1.0
# A gap between DiT calls longer than this is a stall or pause, not the rate.
KEY_INTERVAL_MAX_S = 3.0
WARMUP_SHARE_END = 0.95

# Track Lab skeleton joints (keypoint slot -> name), as skeleton.py writes them.
_LAB_SKELETON_NAMES = {
    31: "neck",
    32: "right_shoulder",
    33: "right_elbow",
    34: "left_shoulder",
    35: "left_elbow",
    36: "chest",
}


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


def gpu_used_fraction(engine: Any | None = None) -> float | None:
    """Used / total CUDA memory, or None when CUDA is unavailable."""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    device = getattr(engine, "device", None) if engine is not None else None
    try:
        if device is not None and getattr(device, "type", "") != "cuda":
            return None
        idx = int(device.index) if device is not None and device.index is not None else 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        total = float(total_b)
        if total <= 0:
            return None
        return max(0.0, min(1.0, 1.0 - (float(free_b) / total)))
    except Exception:
        return None


def should_auto_batch2(used_frac: float | None) -> bool:
    """True when a probe generate left enough GPU headroom for Batch×2."""
    return used_frac is not None and used_frac < GPU_BATCH2_UTIL_LIMIT


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _runtime_log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        path = data_dir() / "vtm_noble.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _image_to_jpeg_b64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def lab_keeps_authored(probe: dict[str, Any] | None) -> bool:
    """True when Track Lab already has rest or mouth end-shapes. Do not wipe."""
    from .blendshapes import SHAPE_IDS

    if not isinstance(probe, dict):
        return False
    shapes = probe.get("shapes") if isinstance(probe.get("shapes"), dict) else {}
    if any(name in shapes for name in SHAPE_IDS):
        return True
    return bool(probe.get("ready"))


def stills_match(incoming: Image.Image | None, path: Path) -> bool:
    if incoming is None or not path.is_file():
        return False
    try:
        with Image.open(path) as cur:
            a = np.asarray(incoming.convert("RGB"))
            b = np.asarray(cur.convert("RGB"))
    except Exception:
        return False
    return a.shape == b.shape and bool(np.array_equal(a, b))


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
        # Webcam loop never runs the hair model — that Mask2Former on CPU
        # was the live FPS cliff. Character hair is captured once on the ref.
        self.tracker = LivePoserTracker(use_hair=False)
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._status = {
            "state": "idle",
            # Launch splash loads the model, last character, and Track Lab.
            "message": "Starting…",
            "boot": new_boot_state(),
            "checkpoint": checkpoint_label(default_ckpt),
            # Picked in the Model list but not loaded yet; Start stream loads it.
            "pending_checkpoint": "",
            "keypoint_layout": str(
                getattr(self.engine, "keypoint_layout", "hrnet_native") or "hrnet_native"
            ),
            "device": "",
            "error": "",
            "model_ready": False,
            "ref_ready": False,
            "streaming": False,
            "tracking": False,
            "track_busy": False,
            "busy": False,
            "models_on_gpu": False,
            "fast_warming": False,
            "steps": STREAM_DEFAULT_STEPS,
            "pose_cfg": STREAM_DEFAULT_POSE_CFG,
            "id_cfg": STREAM_DEFAULT_ID_CFG,
            "frame_blend": STREAM_TEMPORAL_EMA,
            "inbetweens": STREAM_INBETWEENS,
            "interpolate": STREAM_INTERPOLATE,
            "max_fps": STREAM_MAX_GEN_FPS,
            "hold_last": STREAM_HOLD_LAST,
            "paused": False,
            "track_fps": 2.0,
            "drive_pose": True,
            "show_mesh": False,
            "show_hair": False,
            "show_outline": False,
            "show_brows": False,
            "show_eyes": False,
            "show_nose": False,
            "show_mouth": False,
            "show_iris_overlay": False,
            "show_skeleton": False,
            "show_limiters": False,
            "mirror": False,
            "use_iris": True,
            "use_body": True,
            "fast_mode": True,
            "compile_model": bool(getattr(self.engine, "compile_model", STREAM_COMPILE_MODEL)),
            "batch": 0,
            "auto_sync_track": True,
            "gen_fps": 0.0,
            "show_fps": 0.0,
            "timing": "",
            "reference_name": "",
            "reference_path": "",
            "character_id": "",
            "character_name": "",
            "camera_index": 0,
            "track_message": "Tracking off",
            "body_label": "",
            "progress": 0.0,
            "progress_label": "",
            "progress_kind": "",
            "compile_on": False,
            "compile_status": "off",
            "compile_detail": "",
            "virtual_cam": False,
            "virtual_cam_device": "",
            "virtual_cam_error": "",
            "virtual_cam_width": 0,
            "virtual_cam_height": 0,
            "travel_box": default_travel_box(),
            "pose_frozen": False,
            "pose_key_count": 0,
        }
        if not DEVELOPER:
            self._status.update(_PRODUCT_LOCKED_SETTINGS)
        self._boot = new_boot_state()
        self._boot_lock = threading.Lock()
        self._boot_thread: threading.Thread | None = None
        self._boot_character_gate = threading.Event()
        self._status["boot"] = snapshot_boot(self._boot)
        self._ref_path: Path | None = None
        self._last_image: Image.Image | None = None
        self._inbetween_prev: Image.Image | None = None
        self._inbetween_prev_kps: np.ndarray | None = None
        self._restore_last_reference()
        from .ui_session import save_ui_session

        save_ui_session(checkpoint=display_path(default_ckpt))
        self._last_overlay_kps: np.ndarray | None = None
        self._travel_preview_kps: np.ndarray | None = None
        # After a desk edit, live frames must not put the old box back on the slider.
        self._travel_from_desk = False
        self._driven_keypoints: np.ndarray | None = None
        self._last_good_keypoints: np.ndarray | None = None
        self._mesh_edited = False
        self._pose_frozen = False
        self._pose_keys: list = []
        self._freeze_auto_kps: np.ndarray | None = None
        self._freeze_params = None
        self._freeze_roll = 0.0
        self._drag_kp_idx: int | None = None
        self._drag_base_kps: np.ndarray | None = None
        self._drag_slots: set[int] = set()
        self._drag_xy: tuple[float, float] | None = None
        self._last_mesh_emit_t: float = 0.0
        self._streaming = False
        self._paused = False
        self._frame_in_flight = False
        self._last_gen_start = 0.0
        self._gen_hold_pending = False
        self._tracking = False
        self._lab_drive = False
        self._lab_seen_online = False
        self._lab_image_wh: tuple[int, int] | None = None
        # None = this still is not in Track Lab yet; ignore leftover overlay/hair.
        self._lab_overlay_gen: int | None = 0
        self._lab_seen_generation: int = 0
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
        self._prev_controls = None
        self._last_live_preview_t: float = 0.0
        self._prev_stream_kps: np.ndarray | None = None
        self._prev_stream_hair: np.ndarray | None = None
        self._hair_overlay_tracker = None
        self._hair_overlay_lock = threading.Lock()
        self._hair_rig: HairRig | None = None
        self._hair_capture_done: bool = False
        self._last_lab_hair: list | None = None
        self._lab_rest_hair: list | None = None
        self._ema_frame: np.ndarray | None = None
        self._ema_kps: np.ndarray | None = None
        self._fast_warmed: bool = False
        self._batch_picked: bool = False
        # {batch: seconds per call} measured on this PC (hw_profile).
        self._batch_rates: dict[int, float] = {}
        # (batch, seconds) of live stream calls, folded into the profile on Stop.
        self._live_calls: list[tuple[int, float]] = []
        self._gen_busy: bool = False
        self._offload_pending: bool = False
        # Start stream keeps its bar up until the first picture is on screen.
        self._first_frame_pending = False
        # How long each load / warmup stage took on this PC (drives the bars).
        self._stage_clock = StageClock()
        self._vcam_wanted: bool = False
        self._gen_queue: queue.Queue = queue.Queue(maxsize=1)
        self._display_queue: queue.Queue = queue.Queue()
        self._display_busy = False
        self._playout_next = 0.0
        self._last_interp_s = 0.0
        self._last_call_done_t = 0.0
        self._key_interval = 0.0
        self._shown_times: deque[float] = deque(maxlen=128)
        self._model_load_lock = threading.Lock()
        self._pending_checkpoint: Path | None = None
        self._worker_stop = threading.Event()
        self._gen_worker = threading.Thread(
            target=self._gen_worker_loop, name="rs-gen", daemon=True
        )
        self._gen_worker.start()
        self._display_worker = threading.Thread(
            target=self._display_worker_loop, name="rs-interp", daemon=True
        )
        self._display_worker.start()
        self._track_stop = threading.Event()
        self._track_thread = threading.Thread(
            target=self._track_poll_loop, name="rs-track", daemon=True
        )
        self._track_thread.start()

    def _restore_last_reference(self) -> None:
        from .ui_session import load_ui_session

        session = load_ui_session()
        self._status["travel_box"] = normalize_travel_box(session.get("travel_box"))
        try:
            steps = max(1, min(50, int(session.get("steps") or STREAM_DEFAULT_STEPS)))
        except (TypeError, ValueError):
            steps = STREAM_DEFAULT_STEPS
        try:
            pose_cfg = _clip_cfg(float(session.get("pose_cfg") or STREAM_DEFAULT_POSE_CFG))
        except (TypeError, ValueError):
            pose_cfg = STREAM_DEFAULT_POSE_CFG
        try:
            id_cfg = _clip_cfg(float(session.get("id_cfg") or STREAM_DEFAULT_ID_CFG))
        except (TypeError, ValueError):
            id_cfg = STREAM_DEFAULT_ID_CFG
        try:
            frame_blend = _clip_blend(
                float(session.get("frame_blend") or STREAM_TEMPORAL_EMA)
            )
        except (TypeError, ValueError):
            frame_blend = STREAM_TEMPORAL_EMA
        saved_index, _saved_name = _remembered_camera()
        if saved_index is not None:
            self._status["camera_index"] = int(saved_index)
        self._status["steps"] = steps
        self._status["pose_cfg"] = pose_cfg
        self._status["id_cfg"] = id_cfg
        self._status["frame_blend"] = frame_blend
        raw_hold = session.get("hold_last")
        hold_last = STREAM_HOLD_LAST if raw_hold is None else bool(raw_hold)
        self._status["hold_last"] = hold_last
        raw_inb = session.get("inbetweens")
        self._status["inbetweens"] = (
            STREAM_INBETWEENS if raw_inb is None else _clip_inbetweens(raw_inb)
        )
        raw_interp = session.get("interpolate")
        self._status["interpolate"] = (
            STREAM_INTERPOLATE if raw_interp is None else interpolate_on(raw_interp)
        )
        raw_cap = session.get("max_fps")
        self._status["max_fps"] = STREAM_MAX_GEN_FPS if raw_cap is None else _clip_max_fps(raw_cap)
        self._status["batch"] = _clip_batch(session.get("batch"))
        self.engine.set_hold_last(hold_last)
        self.engine.num_steps = steps
        self.engine.set_guidance(pose_cfg=pose_cfg, id_cfg=id_cfg)
        raw_compile = session.get("compile_model")
        if raw_compile is not None:
            compile_model = bool(raw_compile)
            self.engine.set_compile_model(compile_model)
            self._status["compile_model"] = compile_model
        raw = str(session.get("character_path") or session.get("reference_path") or "").strip()
        if not raw:
            return
        try:
            characters_dir()
            ref = resolve_user_path(raw)
        except Exception:
            return
        if not ref.is_file():
            from .ui_session import save_ui_session

            save_ui_session(character_path="", reference_path="")
            return
        self._ref_path = ref
        self._status["reference_name"] = ref.stem if ref.suffix.lower() == ".vtm" else ref.name
        self._status["reference_path"] = display_path(ref)
        if ref.suffix.lower() == ".vtm":
            try:
                self._show_character_still(ref)
            except Exception:
                self._status["character_id"] = ref.stem
                self._status["character_name"] = ref.stem
                self._status["message"] = f"Last character: {ref.stem}"
        else:
            self._status["message"] = f"Last reference: {ref.name}"

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

    def _generate_active(self) -> bool:
        return bool(self._streaming or self._frame_in_flight or self._gen_busy)

    def _set_track_status(self, **kwargs: Any) -> None:
        """Tracking status must not clear Generate / Stream busy."""
        if self._generate_active():
            kwargs.pop("busy", None)
        self._set_status(**kwargs)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._status.update(self._compile_status_fields())
            snap = dict(self._status)
        snap["boot"] = self.boot_snapshot()
        snap["gen_cap"] = gen_cap(
            snap.get("max_fps"), snap.get("interpolate"), snap.get("inbetweens")
        )
        snap["gpu_util"] = self._gpu_util() if getattr(self, "_streaming", False) else None
        snap["batch_rates"] = self._batch_rate_view()
        return snap

    def _batch_rate_view(self) -> dict[str, dict[str, Any]]:
        """Keys/s per batch size on this PC: measured, or predicted from them."""
        from .hw_profile import keys_per_s

        rates = dict(getattr(self, "_batch_rates", {}) or {})
        out: dict[str, dict[str, Any]] = {}
        for b in range(1, STREAM_BATCH_MAX + 1):
            fps = keys_per_s(rates, b)
            if fps:
                out[str(b)] = {"fps": round(fps, 1), "measured": b in rates}
        return out

    def _gpu_util(self) -> int | None:
        """Whole-card GPU busy % for the readout (only while streaming, so
        status polls never wake CUDA just to name the card)."""
        from .gpu_monitor import device_uuid, gpu_utilization

        uuid = getattr(self, "_gpu_uuid", None)
        if uuid is None:
            uuid = self._gpu_uuid = device_uuid(getattr(self.engine, "device", None))
        return gpu_utilization(uuid)

    def boot_snapshot(self) -> dict[str, Any]:
        with self._boot_lock:
            return snapshot_boot(self._boot)

    def _publish_boot(self, **status_kwargs: Any) -> None:
        snap = self.boot_snapshot()
        self._set_status(boot=snap, **status_kwargs)

    def _set_boot_stage(
        self,
        key: str,
        *,
        stage_state: str,
        progress: float | None = None,
        label: str | None = None,
        error: str | None = None,
        **status_kwargs: Any,
    ) -> None:
        with self._boot_lock:
            patch_boot_stage(
                self._boot,
                key,
                stage_state=stage_state,
                progress=progress,
                label=label,
            )
            if error is not None:
                self._boot["error"] = str(error)
        payload = dict(status_kwargs)
        if label is not None:
            payload.setdefault("message", label)
            payload.setdefault("progress_label", label)
            payload.setdefault("progress_kind", key)
        if progress is not None:
            payload.setdefault("progress", progress)
        if error:
            payload.setdefault("error", error)
        self._publish_boot(**payload)

    def start_boot(self) -> dict[str, Any]:
        """Idempotent launch: model, last character, lab handshake. No tracking."""
        with self._boot_lock:
            if self._boot["ready"]:
                return snapshot_boot(self._boot)
            if self._boot["running"]:
                return snapshot_boot(self._boot)
            self._boot["running"] = True
            self._boot["error"] = ""
        self._publish_boot(busy=True, message="Starting…")
        self._boot_thread = threading.Thread(
            target=self._run_boot, name="vtm-boot", daemon=True
        )
        self._boot_thread.start()
        return self.boot_snapshot()

    def _run_boot(self) -> None:
        lab_thread = threading.Thread(target=self._boot_lab, name="vtm-boot-lab", daemon=True)
        lab_thread.start()
        try:
            self._boot_model()
            self._boot_character()
        except Exception as exc:
            with self._boot_lock:
                self._boot["error"] = str(exc)
            self._publish_boot(error=str(exc), message="Startup failed")
        lab_thread.join()
        self._boot_lab_source()
        with self._boot_lock:
            self._boot["running"] = False
            self._boot["ready"] = True
        # The last boot stage left progress at 100%. A later Create / Load reads
        # the bar from here, so a stale value would pin it at 100% the whole time.
        self._publish_boot(
            busy=False, message="Ready", progress=0.0, progress_label="", progress_kind=""
        )

    def _boot_model(self) -> None:
        self._set_boot_stage(
            "model",
            stage_state="run",
            progress=0.04,
            label="Loading model",
            busy=True,
        )
        try:
            self.ensure_model(keep_busy=True)
            self._set_boot_stage(
                "model",
                stage_state="done",
                progress=1.0,
                label="Model ready",
            )
        except Exception as exc:
            self._set_boot_stage(
                "model",
                stage_state="error",
                progress=1.0,
                label="Model failed",
                error=str(exc),
            )

    def _boot_character(self) -> None:
        from .character_pack import CharacterPackError
        from .ui_session import load_ui_session

        kind, ident = previous_load_target(load_ui_session())
        with self._boot_lock:
            self._boot["awaiting"] = ""
            self._boot["suggested"] = ident if kind == "character" else ""
        if kind == "none":
            self._set_boot_stage(
                "character",
                stage_state="skip",
                progress=1.0,
                label="No character",
                error="",
            )
            return
        self._set_boot_stage(
            "character",
            stage_state="run",
            progress=0.12,
            label="Loading character",
            busy=True,
        )
        try:
            if kind == "character":
                self.load_character(ident, replace_lab=False)
            else:
                self.apply_reference(ident, silent=True, warmup=False)
            name = str(self.status().get("character_name") or ident)
            self._set_boot_stage(
                "character",
                stage_state="done",
                progress=1.0,
                label=f"Character: {name}",
            )
        except (CharacterPackError, FileNotFoundError):
            self._forget_character(message="Ready")
            self._set_boot_stage(
                "character",
                stage_state="skip",
                progress=1.0,
                label="No character",
                error="",
                message="Ready",
            )
        except Exception as exc:
            self._set_boot_stage(
                "character",
                stage_state="error",
                progress=1.0,
                label="Character failed",
                error=str(exc),
            )

    def finish_boot_character(self, ident: str = "", *, already_loaded: bool = False) -> dict[str, Any]:
        """Splash Select / Create. Does not start tracking."""
        ident = str(ident or "").strip()
        with self._boot_lock:
            waiting = self._boot.get("awaiting") == "character" and not self._boot.get("ready")
        if ident and not already_loaded:
            self._set_boot_stage(
                "character",
                stage_state="run",
                progress=0.45,
                label="Loading character",
                busy=True,
            )
            self.load_character(ident, require_compatible=False)
        if waiting:
            self._boot_character_gate.set()
        return self.status()

    def _boot_lab(self) -> None:
        self._set_boot_stage(
            "lab",
            stage_state="run",
            progress=0.08,
            label="Connecting Track Lab",
            busy=True,
        )
        from .lab_process import BOOT_WAIT, connect_lab

        def _lab_wait(frac: float) -> None:
            self._set_boot_stage(
                "lab",
                stage_state="run",
                progress=0.1 + 0.8 * max(0.0, min(1.0, float(frac))),
                label="Connecting Track Lab",
                busy=True,
            )

        packet = connect_lab(timeout=float(BOOT_WAIT), on_wait=_lab_wait)
        _runtime_log(
            f"boot lab online={bool(packet.get('online'))} "
            f"loaded={packet.get('loaded')} error={packet.get('error') or '—'}"
        )
        if packet.get("online"):
            from .lab_harness import lab as lab_harness

            self._lab_seen_online = True
            self.adopt_lab_travel_box(packet)
            if packet.get("live") and packet.get("loaded", True):
                try:
                    lab_harness.command("stop")
                except Exception:
                    pass
            self._set_boot_stage(
                "lab",
                stage_state="done",
                progress=1.0,
                label="Track Lab connected",
            )
            return
        err = str(packet.get("error") or "Track Lab is still starting")
        self._set_boot_stage(
            "lab",
            stage_state="run",
            progress=0.7,
            label="Track Lab still starting",
        )
        _runtime_log(f"boot lab not ready yet: {err}")

    def _boot_lab_source(self) -> None:
        """Push the character still and fit rest after handshake + character load."""
        try:
            self._sync_lab_character()
        except Exception:
            return
        self._restore_character_fit()

    def _restore_character_fit(self) -> None:
        """A lab re-track re-detects hair and skeleton. Put the painted ones back."""
        # Track Lab was offline when the character loaded; hand it the limiters now.
        self._apply_character_limiters()
        try:
            changed = self._apply_character_fit()
        except Exception as exc:
            print(f"Character fit restore failed: {exc}")
            return
        if changed and self._last_image is not None:
            self._emit(
                {
                    "type": "frame",
                    **self._frame_payload(self._last_image, self._last_overlay_kps),
                }
            )

    def _sync_lab_character(self, *, replace: bool = False) -> None:
        """Load the current character still into Track Lab and fit the rest mesh.

        Track Lab is the authoring tool. Boot / Start tracking must not wipe
        mouth end-shapes. A new desk character still may replace them.
        """
        from .lab_harness import lab as lab_harness

        probe = lab_harness.status(merge_frame=False)
        if not probe.get("online"):
            return
        self._lab_seen_online = True
        self._note_lab_generation(probe)
        same = self._same_lab_still()
        keep = lab_keeps_authored(probe) and not replace and same
        if keep:
            self._accept_lab_overlay(probe)
            packet = self._lab_packet_from_ack({"status": probe})
            self._adopt_lab_hair(packet)
            self.adopt_lab_overlay(packet, emit=True)
            return
        path = self._write_lab_source()
        if path is None:
            return
        # A .vtm already carries its rest mesh; read it before set_source
        # clears the lab so loading never re-runs the detection models.
        rest = self._packaged_lab_rest(probe)
        prev_gen = int(getattr(self, "_lab_seen_generation", 0) or 0)
        src_ack = lab_harness.put_source(str(path))
        nested = src_ack.get("status") if isinstance(src_ack.get("status"), dict) else {}
        err = str(src_ack.get("error") or nested.get("error") or "")
        if src_ack.get("ok") is False:
            raise RuntimeError(err or "Track Lab could not load the character")
        src_packet = self._lab_packet_from_ack(src_ack)
        src_gen = self._lab_generation(src_packet)
        if prev_gen and src_gen and src_gen <= prev_gen:
            raise RuntimeError("Track Lab did not load the new character still")
        self._accept_lab_overlay(src_packet or src_ack)
        if rest is not None:
            track_ack = self._lab_ack("set_rest", rest)
        else:
            track_ack = self._lab_ack("track")
        packet = self._lab_packet_from_ack(track_ack)
        self._accept_lab_overlay(packet or track_ack)
        got_hair = self._adopt_lab_hair(packet)
        self.adopt_lab_overlay(packet, emit=True)
        if rest is not None and "hair" not in rest and got_hair:
            # First load of a pack without hair: keep what the lab found so
            # the next load is fully packaged too.
            self._store_pack_hair()
        overlay = self._last_overlay_kps
        if overlay is not None:
            try:
                self.engine.adopt_ref_keypoints(
                    overlay, persist=False, pose_source="lab"
                )
            except Exception:
                pass
        if not got_hair:
            self._hair_capture_done = False
            self._maybe_capture_hair(rest_keypoints=self._last_overlay_kps)

    def _packaged_lab_rest(self, probe: dict[str, Any]) -> dict[str, Any] | None:
        """The loaded pack's rest mesh as a Track Lab ``set_rest`` body.

        Face, iris and skeleton come from the pack's keypoints; hair from its
        fit when saved. None = not a pack, the lab lacks ``set_rest``, or the
        mesh is unusable, and the caller runs full detection instead.
        """
        from .character_fit import norm_hair_to_pixels, read_character_fit

        ref = getattr(self, "_ref_path", None)
        if ref is None or ref.suffix.lower() != ".vtm":
            return None
        commands = probe.get("commands")
        if isinstance(commands, list) and "set_rest" not in commands:
            return None
        image = self._last_image
        kps = self._last_overlay_kps
        if image is None or kps is None:
            return None
        k = np.asarray(kps, dtype=np.float32)
        if k.ndim != 2 or k.shape[0] < 37 or k.shape[1] < 4:
            return None
        if int(np.count_nonzero(k[:28, 3] >= 0.5)) < 20:
            return None
        w, h = image.size

        def px(i: int) -> list[float]:
            return [
                round((float(k[i, 0]) + 1.0) * 0.5 * w, 3),
                round((float(k[i, 1]) + 1.0) * 0.5 * h, 3),
            ]

        def score(i: int) -> float:
            s = float(k[i, 2])
            return round(s if s >= 0.05 else 1.0, 4)

        body: dict[str, Any] = {
            "points": [px(i) + [round(float(k[i, 2]), 4)] for i in range(28)],
            # Slots 28 / 29 are the two irises, 31-36 the lab skeleton.
            "iris": [
                {"id": i, "x": px(i)[0], "y": px(i)[1], "score": score(i), "visible": True}
                for i in (28, 29)
                if k[i, 3] >= 0.5
            ],
            "skeleton": [
                {"id": i, "name": name, "x": px(i)[0], "y": px(i)[1], "score": score(i)}
                for i, name in _LAB_SKELETON_NAMES.items()
                if k[i, 3] >= 0.5
            ],
        }
        try:
            hair = read_character_fit(ref.stem).get("hair")
        except Exception:
            hair = None
        if isinstance(hair, list) and hair:
            body["hair"] = norm_hair_to_pixels(hair, w, h)
        return body

    def _store_pack_hair(self) -> None:
        """Save lab-detected hair into the loaded pack's fit (first load only)."""
        from .character_fit import update_character_fit

        ref = getattr(self, "_ref_path", None)
        hair = getattr(self, "_last_lab_hair", None)
        if ref is None or ref.suffix.lower() != ".vtm" or not hair:
            return
        try:
            update_character_fit(ref.stem, {"hair": [dict(seg) for seg in hair]})
        except Exception as exc:
            print(f"Could not store hair in {ref.name}: {exc}")

    def _compile_status_fields(self) -> dict[str, Any]:
        """Derive UI compile light from the engine (always fresh)."""
        engine = getattr(self, "engine", None)
        try:
            st = str(getattr(engine, "compile_status", "off") or "off")
        except Exception:
            st = "off"
        try:
            detail = str(getattr(engine, "compile_detail", "") or "")
        except Exception:
            detail = ""
        try:
            layout = str(
                getattr(engine, "keypoint_layout", "hrnet_native") or "hrnet_native"
            )
        except Exception:
            layout = "hrnet_native"
        try:
            wanted = bool(getattr(engine, "compile_model", False))
        except Exception:
            wanted = False
        try:
            batch_size = max(1, int(getattr(engine, "stream_batch_size", 1) or 1))
        except (TypeError, ValueError):
            batch_size = 1
        return {
            "compile_status": st,
            "compile_on": st == "on",
            "compile_model": wanted,
            "compile_detail": detail,
            "keypoint_layout": layout,
            "models_on_gpu": bool(getattr(engine, "_gpu_resident", False)),
            # Poses per DiT call right now (the Batch setting, or what Auto took).
            "batch_size": batch_size,
            "batch2": batch_size > 1,
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
        boot_key = "model" if kind in {"model", "download"} else kind
        if boot_key in {"model", "character", "lab", "reference"}:
            mapped = "character" if boot_key == "reference" else boot_key
            if mapped in {"model", "character", "lab"}:
                with self._boot_lock:
                    if self._boot.get("running"):
                        patch_boot_stage(
                            self._boot,
                            mapped,
                            stage_state="run",
                            progress=frac,
                            label=label,
                        )
                        payload["boot"] = snapshot_boot(self._boot)
        self._set_status(**payload)

    def _clear_progress(self, **kwargs: Any) -> None:
        self._set_status(
            progress=0.0,
            progress_label="",
            progress_kind="",
            **kwargs,
        )

    def _progress_value(self) -> float:
        with self._lock:
            return float(self._status.get("progress") or 0.0)

    @contextmanager
    def _run_progress_band(
        self,
        *,
        kind: str,
        start: float,
        end: float,
        label: str,
        expected_s: float = 10.0,
    ) -> Iterator[None]:
        """Hold a labeled band and trickle forward while the body is silent."""
        stop = threading.Event()
        started = time.monotonic()
        lo = max(0.0, min(1.0, float(start)))
        hi = max(lo, min(1.0, float(end)))
        self._set_progress(max(self._progress_value(), lo), label=label, kind=kind)

        def _poll() -> None:
            span = max(hi - lo, 1e-6)
            while not stop.wait(0.12):
                elapsed = time.monotonic() - started
                trickle = lo + min(0.90, elapsed / max(expected_s, 0.4)) * span
                cap = hi - 0.008
                with self._lock:
                    cur = float(self._status.get("progress") or 0.0)
                    cur_label = str(self._status.get("progress_label") or label)
                nxt = min(cap, max(cur, trickle, lo))
                self._set_progress(nxt, label=cur_label, kind=kind)

        poller = threading.Thread(target=_poll, name="rs-create-progress", daemon=True)
        poller.start()
        try:
            yield
            self._set_progress(max(self._progress_value(), hi), label=label, kind=kind)
        finally:
            stop.set()
            poller.join(timeout=1.0)

    def _reset_character_runtime(self, *, emit_blank: bool = False) -> None:
        """Drop the previous still, overlay, and stream mix before a new Create."""
        self._last_image = None
        self._last_overlay_kps = None
        self._driven_keypoints = None
        self._last_good_keypoints = None
        self._mesh_edited = False
        self._pose_frozen = False
        self._freeze_auto_kps = None
        self._freeze_params = None
        self._freeze_roll = 0.0
        self._drag_kp_idx = None
        self._drag_slots = set()
        self._drag_base_kps = None
        self._drag_xy = None
        self._hair_rig = None
        self._hair_capture_done = False
        self._last_lab_hair = None
        self._lab_rest_hair = None
        self._lab_image_wh = None
        self._lab_overlay_gen = None
        self._inbetween_prev = None
        self._inbetween_prev_kps = None
        self._prev_stream_kps = None
        self._prev_stream_hair = None
        self._ema_frame = None
        self._ema_kps = None
        if emit_blank:
            blank = Image.new("RGB", (16, 16), (8, 8, 8))
            self._emit({"type": "frame", **self._frame_payload(blank, None)})

    def _guess_skip_crop(self, rgb: np.ndarray, path: Path | None = None) -> bool:
        from .engine import looks_like_greenscreen

        in_train = bool(path) and "train_crop" in str(path).replace("\\", "/")
        return not (in_train or looks_like_greenscreen(rgb))

    def _cel_still(
        self,
        image: Image.Image | np.ndarray,
        *,
        skip_crop: bool,
        image_size: int | None = None,
    ) -> Image.Image:
        """Same square canvas create uses, so overlay + mouse share one space."""
        from .ref_pose_fit import reference_display_rgb

        size = int(image_size or getattr(self.engine, "image_size", 0) or 768)
        if hasattr(image, "convert"):
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == size and arr.shape[1] == size:
            return Image.fromarray(arr)
        return Image.fromarray(
            reference_display_rgb(arr, skip_crop=bool(skip_crop), image_size=size)
        )

    def _emit_incoming_still(self, path: Path) -> None:
        try:
            preview = Image.open(path).convert("RGB")
        except Exception:
            preview = Image.new("RGB", (16, 16), (8, 8, 8))
        try:
            arr = np.asarray(preview)
            preview = self._cel_still(arr, skip_crop=self._guess_skip_crop(arr, path))
        except Exception:
            pass
        self._last_image = preview
        self._emit({"type": "frame", **self._frame_payload(preview, None)})

    def _warm_overlay_tools(self, kind: str) -> None:
        from .ref_pose_fit import get_anime_face_detector, get_iris_model, get_skeleton_model

        with self._run_progress_band(
            kind=kind, start=0.04, end=0.12, label="Loading face detector", expected_s=12.0
        ):
            try:
                get_anime_face_detector(device="cpu")
            except Exception as exc:
                print(f"Face detector load failed: {exc}", flush=True)
        with self._run_progress_band(
            kind=kind, start=0.12, end=0.22, label="Loading iris", expected_s=8.0
        ):
            try:
                get_iris_model(device="cpu")
            except Exception as exc:
                print(f"Iris model load failed: {exc}", flush=True)
        with self._run_progress_band(
            kind=kind, start=0.22, end=0.32, label="Loading skeleton", expected_s=8.0
        ):
            try:
                get_skeleton_model(device="cpu")
            except Exception as exc:
                print(f"Skeleton model load failed: {exc}", flush=True)

    def _stage_meter(
        self,
        keys: list[str],
        *,
        kind: str,
        lo: float = 0.0,
        hi: float = 1.0,
        ckpt: Path | None = None,
        label: str = "",
        **status_kwargs: Any,
    ) -> StageMeter:
        """A bar timed against this PC's earlier runs of the same stages.

        With ``ckpt``, the ``dit`` stage also counts real bytes: process RAM
        grows by about the checkpoint's size while it is read.
        """
        gb = 1.0
        inner = None
        if ckpt is not None and ckpt.is_file():
            size = max(ckpt.stat().st_size, 1)
            gb = size / 1e9
            baseline = _process_rss_bytes()

            def inner(key: str) -> float | None:
                if key != "dit" or baseline <= 0:
                    return None
                rss = _process_rss_bytes()
                return None if rss <= 0 else max(0, rss - baseline) / size

        def report(value: float, label: str) -> None:
            self._set_progress(value, label=label, kind=kind, **status_kwargs)

        return StageMeter(
            self._stage_clock, keys, report, lo=lo, hi=hi, gb=gb, inner=inner, label=label
        )

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
                        message="Downloading VTM-1.5.1.pt into models/dit…",
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

            keys = ["dit"] if self.engine.vae is not None else ["dit", "vae"]
            with self._stage_meter(keys, kind="model", ckpt=ckpt, label="Loading model") as meter:
                self.engine.load(on_stage=meter.stage)
                meter.stage("done", "Model ready")
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

    def ensure_model(self, *, keep_busy: bool = False, keep_bar: bool = False) -> None:
        """Load DiT+VAE once, on demand (not at UI open).

        ``keep_bar``: a warmup follows (Start stream, Generate once), so the
        GPU-move slice stays on screen and the warmup bar carries on from it.
        A model picked in the Model list loads here first.
        """
        self._apply_pending_checkpoint()
        with self._model_load_lock:
            if not getattr(self.engine, "_ready", False) or self.engine.model is None:
                self._load_model(clear_busy=not keep_busy)
            # Stop stream moves the weights to RAM; say so while they come back.
            # The first slice of the warmup bar, so Start reads as one bar.
            if not bool(getattr(self.engine, "_gpu_resident", True)):
                was_busy = bool(self.status().get("busy"))
                with self._stage_meter(
                    ["gpu_move"], kind="warmup", hi=GPU_MOVE_SHARE, label="Moving model to GPU"
                ) as meter:
                    meter.stage("gpu_move", "Moving model to GPU")
                    try:
                        self.engine.ensure_gpu()
                    except Exception:
                        pass
                    meter.stage("done", "")
                if not keep_bar:
                    self._clear_progress(busy=was_busy or keep_busy)
            else:
                try:
                    self.engine.ensure_gpu()
                except Exception:
                    pass
            if bool(self.status().get("model_ready")):
                return
            self._set_status(
                state="ready",
                message="Model ready",
                model_ready=True,
                device=str(getattr(self.engine, "device", "")),
            )

    def list_checkpoints(self) -> list[dict[str, str]]:
        from .model_download import hub_checkpoint_names

        hub = hub_checkpoint_names()
        return [
            {
                "label": label,
                "path": display_path(path),
                "source": "hub" if path.name in hub else "local",
            }
            for label, path in list_stream_checkpoints()
        ]

    def _reload_character_after_model(self) -> None:
        """Put the open pack's latents back on the new DiT. Do not reopen the zip as a picture or resync Track Lab."""
        ref = self._ref_path
        if ref is None or ref.suffix.lower() != ".vtm" or not ref.is_file():
            return
        from .character_pack import read_character_pack

        pack = read_character_pack(ref)
        if self._pack_latents_usable(pack):
            self.engine.load_encoded_reference(
                keypoints=pack.keypoints,
                ref_latent=pack.ref_latent,
                ref_face_latent=pack.ref_face_latent,
                skip_crop=pack.skip_crop,
                path=ref,
            )
            return
        from .paths import refs_dir

        if pack.source_bytes:
            tmp = refs_dir() / f"_character_fallback{pack.source_suffix or '.png'}"
            tmp.write_bytes(pack.source_bytes)
        else:
            tmp = refs_dir() / "_character_fallback.png"
            Image.fromarray(pack.preview_rgb).save(tmp)
        try:
            self.engine.set_reference(tmp, pack.keypoints, skip_crop=pack.skip_crop)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
            discard_sidecar_keypoints(tmp)
        self.engine._ref_path = ref
        self._store_reencoded_latents(ref)

    @staticmethod
    def _checked_checkpoint(path: str | Path) -> Path:
        ckpt = resolve_user_path(path).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {display_path(ckpt)}")
        if not is_stream_checkpoint_file(ckpt):
            raise ValueError(
                f"Not a DiT checkpoint (need a .pt/.pth file over 1 MB, got {display_path(ckpt)})"
            )
        return ckpt

    def select_checkpoint(self, path: str | Path) -> None:
        """Pick a model without loading it. Start stream / Generate load it.

        Swapping weights runs the whole load + warmup, so the Model list only
        records the choice and the Start button asks for it.
        """
        ckpt = self._checked_checkpoint(path)
        remember_checkpoint_location(ckpt)
        loaded = Path(self.engine.checkpoint).resolve()
        same = ckpt == loaded and bool(getattr(self.engine, "_ready", False))
        with self._lock:
            self._pending_checkpoint = None if same else ckpt
        from .ui_session import save_ui_session

        # The next launch boots the picked model either way.
        save_ui_session(checkpoint=display_path(ckpt))
        if same:
            self._set_status(pending_checkpoint="", message="Model ready", error="")
        else:
            self._set_status(
                pending_checkpoint=checkpoint_label(ckpt),
                message="New model selected — press Start stream to load it",
                error="",
            )

    def _apply_pending_checkpoint(self) -> None:
        with self._lock:
            pending = self._pending_checkpoint
        if pending is not None:
            self.set_checkpoint(pending)

    def set_checkpoint(self, path: str | Path) -> None:
        ckpt = self._checked_checkpoint(path)
        remember_checkpoint_location(ckpt)
        with self._lock:
            self._pending_checkpoint = None
        with self._model_load_lock:
            self._set_status(busy=True, message="Switching model…", error="")
            try:
                has_pack = self._ref_path is not None and self._ref_path.suffix.lower() == ".vtm"
                keys = ["dit"]
                if self.engine.vae is None:
                    keys.append("vae")
                if has_pack:
                    keys.append("character")
                with self._stage_meter(
                    keys, kind="model", ckpt=ckpt, label="Loading model"
                ) as meter:
                    self.engine.set_checkpoint(ckpt, on_stage=meter.stage)
                    if has_pack:
                        # The open character goes onto the new model (a
                        # re-encode when its latents came from another one).
                        meter.stage("character", "Loading character on the new model")
                    self._reload_character_after_model()
                    meter.stage("done", "Model ready")
                self._fast_warmed = False
                self._batch_picked = False
                try:
                    self.engine.set_stream_batch_size(1)
                except Exception:
                    pass
                self._clear_progress(
                    busy=False,
                    checkpoint=checkpoint_label(self.engine.checkpoint),
                    pending_checkpoint="",
                    message="Model ready",
                    model_ready=True,
                    state="ready",
                    error="",
                )
                from .ui_session import save_ui_session

                save_ui_session(checkpoint=display_path(ckpt))
            except Exception as exc:
                _runtime_log("model switch failed:\n" + traceback.format_exc())
                ready = bool(getattr(self.engine, "_ready", False))
                self._clear_progress(
                    busy=False,
                    error=str(exc),
                    message="Model switch failed",
                    model_ready=ready,
                    checkpoint=checkpoint_label(self.engine.checkpoint),
                    pending_checkpoint="",
                )
                raise

    def apply_reference(
        self,
        path: str | Path,
        *,
        silent: bool = False,
        warmup: bool = True,
    ) -> dict[str, Any]:
        ref = resolve_user_path(path)
        if not ref.is_file():
            raise FileNotFoundError(f"Reference not found: {ref}")
        if ref.suffix.lower() == ".vtm":
            # A character pack is a zip. Load its preview and latents.
            loaded = self.load_character(ref.stem, require_compatible=False)
            frame = loaded.get("frame") if isinstance(loaded, dict) else None
            return frame if isinstance(frame, dict) else {}
        if self._streaming:
            raise RuntimeError("Stop the stream before applying a new reference")
        self._set_status(
            busy=True,
            message="Creating character…" if silent else "Applying reference…",
            error="",
            progress=0.0,
            progress_label="",
            progress_kind="",
        )
        try:
            # Load DiT+VAE first so status can say "Loading…" instead of looking stuck.
            if not getattr(self.engine, "_ready", False):
                self.ensure_model(keep_busy=True)

            kind = "character" if silent else "reference"

            def _on_ref_progress(frac: float, label: str) -> None:
                raw = float(frac)
                text = str(label or "")
                if silent:
                    low = text.lower()
                    if "fit" in low or "pose" in low or raw < 0.50:
                        mapped = 0.32 + min(raw, 0.55) / 0.55 * 0.16
                        text = "Fitting pose"
                    else:
                        mapped = 0.48 + max(0.0, raw - 0.55) / 0.45 * 0.24
                        text = "Encoding character"
                    self._set_progress(max(self._progress_value(), mapped), label=text, kind=kind)
                    return
                self._set_progress(raw, label=text, kind=kind)

            if silent:
                self._reset_character_runtime(emit_blank=True)
                discard_sidecar_keypoints(ref)
                self._emit_incoming_still(ref)
                self._warm_overlay_tools(kind)
            else:
                self._set_progress(
                    0.05,
                    label="Encoding reference…",
                    kind=kind,
                )
            pose_band = None
            if silent:
                pose_band = self._run_progress_band(
                    kind=kind,
                    start=0.32,
                    end=0.72,
                    label="Fitting pose",
                    expected_s=22.0,
                )
            if pose_band is not None:
                with pose_band:
                    self.engine.set_reference(ref, on_progress=_on_ref_progress)
            else:
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
            self._pose_frozen = False
            self._freeze_auto_kps = None
            self._drag_kp_idx = None
            self._drag_slots = set()
            self._drag_base_kps = None
            self._drag_xy = None
            self._hair_rig = None
            self._hair_capture_done = False
            self._last_lab_hair = None
            self._lab_rest_hair = None
            self._lab_image_wh = None
            self._reset_live_origin(reason="apply_reference")
            preview = None
            try:
                arr = np.asarray(Image.open(ref).convert("RGB"))
                skip_crop = bool(getattr(self.engine, "_ref_skip_crop", False))
                preview = self._cel_still(arr, skip_crop=skip_crop)
                self._last_image = preview
            except Exception:
                preview = Image.open(ref).convert("RGB")
                self._last_image = preview
            overlay_lo = 0.72 if silent else 0.90
            overlay_hi = 0.86 if silent else 0.96
            try:
                with self._run_progress_band(
                    kind=kind,
                    start=overlay_lo,
                    end=overlay_hi,
                    label="Fitting overlay…",
                    expected_s=10.0,
                ):
                    self._sync_lab_character(replace=True)
            except Exception as exc:
                print(f"lab character sync failed: {exc}")
            self._maybe_capture_hair(rest_keypoints=self._last_overlay_kps)
            frame = self._frame_payload(preview, self._last_overlay_kps)
            if silent:
                self._set_status(
                    busy=True,
                    ref_ready=True,
                    reference_name=ref.name,
                    reference_path=display_path(ref),
                    message="Creating character…",
                    error="",
                )
            else:
                self._clear_progress(
                    busy=True,
                    ref_ready=True,
                    reference_name=ref.name,
                    reference_path=display_path(ref),
                    message="Reference applied",
                    error="",
                )
            self._emit({"type": "frame", **frame})
            from .ui_session import save_ui_session

            save_ui_session(reference_path=display_path(ref))
            self._reload_pose_keys(ref)
            # Fast path: compile + warmup so the first stream frame isn't a freeze.
            # Character-pack create/load skips this; Stream still compiles later.
            if warmup and not silent:
                self._run_fast_warmup_if_needed()
            if not silent:
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

    def list_characters(self) -> list[dict[str, Any]]:
        from .character_pack import (
            FORMAT_VERSION,
            ensure_character_still,
            list_character_files,
            peek_character_manifest,
        )

        self._refresh_blend_current()
        cards: list[dict[str, Any]] = []
        for path in list_character_files():
            try:
                # One-time: older packs take in their sidecars and become v2.
                if int(peek_character_manifest(path).get("version") or 0) < FORMAT_VERSION:
                    self._migrate_character_pack(path)
                ensure_character_still(path)
                cards.append(self._character_card_safe(path))
            except Exception:
                cards.append(
                    {
                        "id": path.stem,
                        "name": path.stem,
                        "path": display_path(path),
                        "preview_url": f"/api/characters/{path.stem}/preview?v={int(path.stat().st_mtime_ns)}",
                    }
                )
        return cards

    def _character_path(self) -> Path | None:
        path = self._ref_path
        return path if path is not None and path.is_file() else None

    def current_character_path(self) -> Path | None:
        path = self._character_path()
        return path if path is not None and path.suffix.lower() == ".vtm" else None

    def _show_character_still(self, path: Path) -> None:
        from .character_pack import (
            CharacterPackError,
            ensure_character_still,
            peek_character_manifest,
        )

        try:
            raw = peek_character_manifest(path)
            name = str(raw.get("name") or path.stem)
        except CharacterPackError:
            name = path.stem
        still = ensure_character_still(path)
        preview = Image.open(still).convert("RGB")
        self._last_image = preview
        frame = self._frame_payload(preview, None)
        self._emit({"type": "frame", **frame})
        self._status["character_id"] = path.stem
        self._status["character_name"] = name
        self._status["reference_name"] = name
        self._status["reference_path"] = display_path(path)
        self._status["message"] = f"Character: {name}"

    def _mark_current_character(self, path: Path, name: str) -> None:
        from .ui_session import save_ui_session

        self._ref_path = path
        ident = path.stem
        self._set_status(
            ref_ready=True,
            reference_name=name,
            reference_path=display_path(path),
            character_id=ident,
            character_name=name,
            message=f"Character: {name}",
            error="",
        )
        save_ui_session(
            character_path=display_path(path),
            reference_path=display_path(path),
        )
        self._reload_pose_keys(path)

    def _reload_pose_keys(self, path: Path | None) -> None:
        keys = load_keys(path)
        self._pose_keys = keys
        self._set_status(pose_key_count=len(keys), pose_frozen=bool(self._pose_frozen))

    def _persist_pose_keys(self) -> None:
        save_keys(self._ref_path, self._pose_keys)
        self._set_status(pose_key_count=len(self._pose_keys))

    def _install_loaded_reference(self, preview: Image.Image, kps: np.ndarray) -> dict[str, Any]:
        self._last_overlay_kps = np.asarray(kps, dtype=np.float32).copy()
        self._driven_keypoints = self._last_overlay_kps.copy()
        self._last_good_keypoints = self._last_overlay_kps.copy()
        self._mesh_edited = False
        self._pose_frozen = False
        self._freeze_auto_kps = None
        self._drag_kp_idx = None
        self._drag_slots = set()
        self._drag_base_kps = None
        self._drag_xy = None
        self._hair_rig = None
        self._hair_capture_done = False
        self._last_lab_hair = None
        self._lab_rest_hair = None
        self._lab_image_wh = None
        self._lab_overlay_gen = None
        self._reset_live_origin(reason="load_character")
        self._last_image = preview
        frame = self._frame_payload(preview, self._last_overlay_kps)
        self._emit({"type": "frame", **frame})
        return frame

    def create_character(self, image_path: str | Path, *, name: str | None = None) -> dict[str, Any]:
        from .character_pack import unique_character_path, write_character_pack

        src = resolve_user_path(image_path)
        if not src.is_file():
            raise FileNotFoundError(f"Reference image not found: {src}")
        label = str(name or src.stem).strip() or src.stem
        # The clean original travels in the pack so another model can re-encode it.
        source_bytes = src.read_bytes()
        try:
            frame = self.apply_reference(src, silent=True, warmup=False)
            exported = self.engine.export_encoded_reference()
            preview = (
                np.asarray(self._last_image.convert("RGB"))
                if self._last_image is not None
                else np.asarray(Image.open(src).convert("RGB"))
            )
            dest = unique_character_path(label)
            with self._run_progress_band(
                kind="character",
                start=0.86,
                end=0.94,
                label="Saving character",
                expected_s=4.0,
            ):
                write_character_pack(
                    dest,
                    name=label,
                    preview_rgb=preview,
                    keypoints=exported["keypoints"],
                    ref_latent=exported["ref_latent"],
                    ref_face_latent=exported["ref_face_latent"],
                    image_size=int(exported["image_size"]),
                    skip_crop=bool(exported["skip_crop"]),
                    source_name=f"{label}{src.suffix.lower()}",
                    fit={"travel_box": normalize_travel_box(self._status.get("travel_box"))},
                    source_bytes=source_bytes,
                    source_suffix=src.suffix or ".png",
                    pose_keys=[],
                    model=exported.get("model"),
                    meta={},
                )
                if getattr(self, "_lab_overlay_gen", None) is not None:
                    self._snapshot_character_shapes(dest.stem)
            self.engine._ref_path = dest
            self._mark_current_character(dest, label)
            self._clear_progress(
                busy=False,
                ref_ready=True,
                message=f"Character: {label}",
                error="",
            )
            return {
                "character": self._character_card_safe(dest),
                "frame": frame,
                "status": self.status(),
            }
        finally:
            if src.name.startswith("character_create_"):
                try:
                    src.unlink()
                except OSError:
                    pass
                discard_sidecar_keypoints(src)

    def add_character(self, pack_path: str | Path, *, dest_dir=None) -> dict[str, Any]:
        from .character_pack import (
            unique_character_path,
            validate_character_pack,
        )
        from .paths import characters_dir

        src = resolve_user_path(pack_path)
        if not src.is_file():
            raise FileNotFoundError(f"Character pack not found: {src}")
        from .blendshapes import mark_plan_imported
        from .character_pack import export_character_pack_bytes

        # Hash-checked read: a tampered or truncated pack is refused here.
        manifest = validate_character_pack(src)
        folder = dest_dir if dest_dir is not None else characters_dir()
        dest = unique_character_path(str(manifest.get("name") or src.stem), dest_dir=folder)
        # v1 imports are stored as v2 so they carry everything from now on.
        dest.write_bytes(export_character_pack_bytes(src))
        try:
            # The creator's blend shapes come with the pack; never "repair" them away.
            mark_plan_imported(dest)
        except Exception:
            pass
        return {
            "character": self._character_card_safe(dest),
            "status": self.status(),
            "characters": self.list_characters(),
        }

    def load_character(
        self,
        ident: str,
        *,
        replace_lab: bool = False,
        repair: bool = False,
        require_compatible: bool = True,
        quiet: bool = False,
    ) -> dict[str, Any]:
        from .character_pack import read_character_pack, resolve_character_id

        if self._streaming and not quiet:
            raise RuntimeError("Stop the stream before loading a character")
        path = resolve_character_id(ident)

        def step(frac: float, label: str) -> None:
            # Real checkpoints only; quiet loads run under another job's bar.
            if not quiet:
                self._set_progress(frac, label=label, kind="character")

        if not quiet:
            self._set_status(busy=True, message="Reading character…", error="")
        step(0.05, "Reading character")
        self._migrate_character_pack(path)
        pack = read_character_pack(path)
        if not quiet:
            self._set_status(busy=True, message=f"Loading {pack.name}…", error="")
        step(0.2, f"Loading {pack.name}")
        try:
            if not getattr(self.engine, "_ready", False):
                self.ensure_model(keep_busy=True)
                step(0.3, f"Loading {pack.name}")
            can_reuse = self._pack_latents_usable(pack)
            step(0.35, "Placing character" if can_reuse else "Encoding character")
            if can_reuse:
                self.engine.load_encoded_reference(
                    keypoints=pack.keypoints,
                    ref_latent=pack.ref_latent,
                    ref_face_latent=pack.ref_face_latent,
                    skip_crop=pack.skip_crop,
                    path=path,
                )
            else:
                # Made with another model (or size): re-encode, preferring the
                # clean original over the processed preview, then store the new
                # latents so the next load is instant.
                from .paths import refs_dir

                if pack.source_bytes:
                    suffix = pack.source_suffix or ".png"
                    tmp = refs_dir() / f"_character_fallback{suffix}"
                    tmp.write_bytes(pack.source_bytes)
                else:
                    tmp = refs_dir() / "_character_fallback.png"
                    Image.fromarray(pack.preview_rgb).save(tmp)
                try:
                    self.engine.set_reference(
                        tmp,
                        pack.keypoints,
                        skip_crop=pack.skip_crop,
                        on_progress=lambda frac, _label: step(
                            0.35 + 0.3 * max(0.0, min(1.0, float(frac))), "Encoding character"
                        ),
                    )
                finally:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    discard_sidecar_keypoints(tmp)
                self.engine._ref_path = path
                self._store_reencoded_latents(path)
            kps = getattr(self.engine, "_ref_keypoints", None)
            if kps is None:
                raise RuntimeError("Character pack did not install a rest pose")
            step(0.68, "Placing character")
            size = int(getattr(self.engine, "image_size", 0) or pack.image_size or 768)
            preview = self._cel_still(
                pack.preview_rgb,
                skip_crop=pack.skip_crop,
                image_size=size,
            )
            frame = self._install_loaded_reference(preview, np.asarray(kps))
            self._mark_current_character(path, pack.name)
            self._apply_character_limiters()
            overlay_band = (
                nullcontext()
                if quiet
                else self._run_progress_band(
                    kind="character", start=0.72, end=0.88, label="Fitting overlay…", expected_s=8.0
                )
            )
            try:
                with overlay_band:
                    self._sync_lab_character(replace=replace_lab)
            except Exception:
                pass
            if not self._last_lab_hair:
                self._hair_capture_done = False
                self._maybe_capture_hair(rest_keypoints=self._last_overlay_kps)
                frame = self._frame_payload(self._last_image, self._last_overlay_kps)
                self._emit({"type": "frame", **frame})
            if self._apply_character_fit():
                frame = self._frame_payload(self._last_image, self._last_overlay_kps)
                self._emit({"type": "frame", **frame})
            # Repair after Track Lab holds this still. Before that, the plan is
            # still rebased onto the previous character's face.
            # A mismatch does not keep the previous character on the desk.
            step(0.9, "Checking blend shapes")
            self._character_shape_gate(
                path.stem,
                repair=repair,
                require_compatible=False,
            )
            if quiet:
                self._set_status(busy=False)
            else:
                self._clear_progress(busy=False)
            result: dict[str, Any] = {
                "character": self._character_card_safe(path),
                "frame": frame,
                "status": self.status(),
            }
            if require_compatible:
                from .blendshapes import INCOMPATIBLE_MESSAGE, compatibility

                info = compatibility(path.stem)
                if not info["compatible"]:
                    result["incompatible"] = True
                    result["message"] = INCOMPATIBLE_MESSAGE
            return result
        except Exception as exc:
            self._clear_progress(busy=False, error=str(exc), message="Character load failed")
            raise

    def _character_is_loaded(self, path: Path) -> bool:
        ident = path.stem
        if str(self._status.get("character_id") or "") == ident:
            return True
        current = self._ref_path
        return current is not None and current.stem == ident

    def _clear_engine_reference(self) -> None:
        eng = self.engine
        for attr in (
            "_ref_path",
            "_ref_kps_path",
            "_ref_latent",
            "_ref_face_latent",
            "_ref_keypoints",
            "_ref_keypoints_model",
            "_ref_keypoints_session_base",
            "_ref_rig",
        ):
            try:
                setattr(eng, attr, None)
            except Exception:
                pass
        clear = getattr(eng, "clear_last_gen_latent", None)
        if callable(clear):
            clear()

    def _forget_character(self, *, message: str = "") -> None:
        from .ui_session import save_ui_session

        self._reset_character_runtime()
        self._ref_path = None
        self._reload_pose_keys(None)
        self._reset_live_origin(reason="forget_character")
        self._clear_engine_reference()
        save_ui_session(character_path="", reference_path="")
        self._set_status(
            character_id="",
            character_name="",
            reference_name="",
            reference_path="",
            ref_ready=False,
            pose_frozen=False,
            pose_key_count=0,
            error="",
            message=message,
        )
        self._emit({"type": "frame", "image": None, "width": 0, "height": 0, "keypoints": None})

    def remove_character(self, ident: str) -> dict[str, Any]:
        from .blendshapes import delete_character_plan
        from .character_pack import delete_character_pack, resolve_character_id

        path = resolve_character_id(ident)
        loaded = self._character_is_loaded(path)
        delete_character_pack(path)
        delete_character_plan(path.stem)
        if loaded:
            self._forget_character(message="Character removed")
        return {"ok": True, "status": self.status(), "characters": self.list_characters()}

    def character_info(self, ident: str) -> dict[str, Any]:
        """What a pack carries, for the sharing panel."""
        from .character_pack import read_character_pack, resolve_character_id

        path = resolve_character_id(ident)
        pack = read_character_pack(path)
        fit = pack.fit if isinstance(pack.fit, dict) else {}
        model = pack.model if isinstance(pack.model, dict) else {}
        plan = pack.blendshapes if isinstance(pack.blendshapes, dict) else {}
        try:
            model_match = self._pack_latents_usable(pack)
        except Exception:
            model_match = False
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        return {
            "id": path.stem,
            "name": pack.name,
            "version": int(pack.version),
            "created_at": pack.created_at,
            "updated_at": pack.updated_at,
            "author": str(pack.meta.get("author") or ""),
            "license": str(pack.meta.get("license") or ""),
            "description": str(pack.meta.get("description") or ""),
            "model": {
                "checkpoint": str(model.get("checkpoint") or ""),
                "image_size": int(model.get("image_size") or pack.image_size or 0),
                "latent_shape": list(model.get("latent_shape") or np.shape(pack.ref_latent)),
            },
            "model_match": bool(model_match),
            "includes": {
                "pose_keys": len(pack.pose_keys or []),
                "blendshapes": bool(plan.get("shapes")),
                "hair": bool(fit.get("hair")),
                "skeleton": bool(fit.get("skeleton")),
                "travel_box": bool(fit.get("travel_box")),
                "source_image": bool(pack.source_bytes),
            },
            "size_bytes": size,
        }

    def update_character_meta(
        self,
        ident: str,
        *,
        name: str | None = None,
        author: str | None = None,
        license: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        from .character_pack import resolve_character_id, update_pack_meta

        path = resolve_character_id(ident)
        if author is not None or license is not None or description is not None:
            update_pack_meta(path, author=author, license=license, description=description)
        if name is not None and name.strip():
            return self.rename_character(path.stem, name)
        return {
            "ok": True,
            "character": self._character_card_safe(path),
            "status": self.status(),
            "characters": self.list_characters(),
        }

    def export_character_bytes(self, ident: str) -> tuple[str, bytes]:
        """``(file name, v2 pack bytes)`` with every sidecar folded in first."""
        from .character_pack import (
            export_character_pack_bytes,
            read_character_pack,
            resolve_character_id,
            slugify_character_name,
        )

        path = resolve_character_id(ident)
        self._migrate_character_pack(path)
        name = read_character_pack(path).name or path.stem
        return f"{slugify_character_name(name)}.vtm", export_character_pack_bytes(path)

    def _rename_character_file(self, path: Path, name: str) -> Path:
        """Name the ``.vtm`` after the character, so the file in the folder is
        the one you'd send. Its preview folder moves with it."""
        from .character_pack import (
            character_still_path,
            slugify_character_name,
            unique_character_path,
        )

        library = characters_dir()
        try:
            if path.parent.resolve() != library.resolve():
                return path  # nested pack folders keep their layout
        except OSError:
            return path
        if slugify_character_name(name).lower() == path.stem.lower():
            return path
        dest = unique_character_path(name, dest_dir=library)
        path.rename(dest)
        old_dir = character_still_path(path.stem, dest_dir=library).parent
        new_dir = character_still_path(dest.stem, dest_dir=library).parent
        if old_dir.is_dir() and not new_dir.exists():
            try:
                old_dir.rename(new_dir)
            except OSError:
                pass
        return dest

    def reveal_character(self, ident: str) -> dict[str, Any]:
        """Open Explorer with the character's ``.vtm`` selected, ready to share."""
        import subprocess

        from .character_pack import resolve_character_id

        path = resolve_character_id(ident)
        # Everything the character needs goes into the file before anyone copies it.
        self._migrate_character_pack(path)
        target = path.resolve()
        if sys.platform == "win32":
            subprocess.Popen(f'explorer /select,"{target}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent)])
        return {"ok": True, "path": display_path(path)}

    def rename_character(self, ident: str, name: str) -> dict[str, Any]:
        from .character_pack import rename_character_pack, resolve_character_id
        from .ui_session import save_ui_session

        path = resolve_character_id(ident)
        # Sidecars are keyed by the old file name: fold them in before it changes.
        self._migrate_character_pack(path)
        card = rename_character_pack(path, name)
        loaded = self._ref_path is not None and self._ref_path.resolve() == path.resolve()
        path = self._rename_character_file(path, card["name"])
        if loaded:
            self._ref_path = path
            self.engine._ref_path = path
            self._set_status(
                character_id=path.stem,
                character_name=card["name"],
                reference_name=card["name"],
                reference_path=display_path(path),
                message=f"Character: {card['name']}",
            )
            save_ui_session(
                character_path=display_path(path),
                reference_path=display_path(path),
            )
        return {
            "ok": True,
            "character": self._character_card_safe(path),
            "status": self.status(),
            "characters": self.list_characters(),
        }

    def _refresh_blend_current(self) -> None:
        """Copy Track Lab authored shapes into models/blendshapes/current.json."""
        from .blendshapes import refresh_current_from_lab
        from .lab_harness import lab as lab_harness

        try:
            packet = lab_harness.status(merge_frame=False)
        except Exception:
            return
        if packet.get("online"):
            refresh_current_from_lab(packet)

    def _snapshot_character_shapes(self, ident: str) -> None:
        """New characters take a copy of the latest plan shapes."""
        from .blendshapes import has_plan, load_current, save_character

        self._refresh_blend_current()
        plan = load_current()
        if has_plan(plan.get("shapes")):
            save_character(ident, plan["shapes"])

    def _character_shape_gate(
        self,
        ident: str,
        *,
        repair: bool,
        require_compatible: bool,
    ) -> dict[str, Any] | None:
        from .blendshapes import (
            INCOMPATIBLE_MESSAGE,
            apply_current_to_character,
            compatibility,
        )
        from .character_pack import resolve_character_id

        self._refresh_blend_current()
        info = compatibility(ident)
        if repair and not info["compatible"]:
            apply_current_to_character(ident)
            return None
        if require_compatible and not info["compatible"]:
            path = resolve_character_id(ident)
            return {
                "ok": False,
                "incompatible": True,
                "message": INCOMPATIBLE_MESSAGE,
                "character": self._character_card_safe(path),
                "status": self.status(),
            }
        return None

    def _migrate_character_pack(self, path: Path) -> None:
        """Fold every legacy sidecar into the pack and make it v2.

        Pose keys, the blend shape snapshot and the fit used to live beside
        the pack; each reader folds its own file in on first read.
        """
        from .blendshapes import load_character_plan
        from .character_fit import read_character_fit
        from .character_pack import upgrade_character_pack

        for fold in (
            lambda: load_keys(path),
            lambda: load_character_plan(path.stem),
            lambda: read_character_fit(path.stem),
        ):
            try:
                fold()
            except Exception:
                pass
        try:
            model = self.engine.model_identity() if getattr(self.engine, "_ready", False) else None
            upgrade_character_pack(path, model=None if model is None else {
                k: v for k, v in model.items() if k != "latent_shape"
            })
        except Exception:
            pass

    def _pack_latents_usable(self, pack: Any) -> bool:
        """Stored latents are reused only for the model and size that made them."""
        from .character_pack import pack_model_matches

        ident = self.engine.model_identity()
        return pack_model_matches(
            pack,
            checkpoint=str(ident.get("checkpoint") or ""),
            image_size=int(ident.get("image_size") or 0),
            # The pack's own latent shape: VAE latents depend on image size,
            # which is compared above; the model tag guards the rest.
            latent_shape=np.shape(pack.ref_latent),
        )

    def _store_reencoded_latents(self, path: Path) -> None:
        from .character_pack import replace_pack_latents

        try:
            exported = self.engine.export_encoded_reference()
            replace_pack_latents(
                path,
                ref_latent=exported["ref_latent"],
                ref_face_latent=exported["ref_face_latent"],
                image_size=int(exported["image_size"]),
                model=exported.get("model"),
            )
        except Exception as exc:
            print(f"Could not store re-encoded latents in {path.name}: {exc}")

    def _character_card_safe(self, path: Path) -> dict[str, Any]:
        from .blendshapes import plan_card_fields
        from .character_pack import character_card

        card = character_card(path)
        try:
            card.update(plan_card_fields(path.stem))
        except Exception:
            pass
        return card

    def _run_fast_warmup_if_needed(self, *, force: bool = False) -> None:
        """Compile + prime Fast kernels with a visible progress bar.

        Blocks the caller until compile/warmup finishes. The bar is timed
        against this PC's earlier warmups (``load_timing``), stage by stage.
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

        # Warming up means a stream is coming; drop a Stop-stream offload that
        # was still waiting for the last frame.
        self._offload_pending = False
        try:
            self.engine.ensure_gpu()
        except Exception:
            pass
        self._pick_batch()
        steps = int(self.status().get("steps") or STREAM_DEFAULT_STEPS)
        # Carry on from the GPU move's slice so Start reads as one bar.
        with self._lock:
            carried = (
                float(self._status.get("progress") or 0.0)
                if self._status.get("progress_kind") == "warmup"
                else 0.0
            )
        lo = max(0.02, min(carried, GPU_MOVE_SHARE))
        self._set_status(busy=True, fast_warming=True, message="Preparing stream", error="")

        def _warm(batch: int, label: str = "Preparing stream") -> None:
            keys = self.engine.warmup_plan(batch_size=batch)
            with self._stage_meter(
                keys,
                kind="warmup",
                lo=lo,
                hi=WARMUP_SHARE_END,
                label=label,
                fast_warming=True,
            ) as meter:
                self.engine.warmup(num_steps=steps, batch_size=batch, on_stage=meter.stage)

        try:
            _warm(int(getattr(self.engine, "stream_batch_size", 1) or 1))
            if self._batch_is_auto():
                self._tune_batch(_warm)
            self._fast_warmed = True
            compile_on = bool(getattr(self.engine, "compile_status", "") == "on")
            msg = "Ready — speed boost on" if compile_on else "Ready"
            batch_n = int(getattr(self.engine, "stream_batch_size", 1) or 1)
            if batch_n > 1:
                used = gpu_used_fraction(self.engine)
                if used is not None:
                    msg = f"{msg} · batch ×{batch_n} (GPU {used:.0%} VRAM)"
                else:
                    msg = f"{msg} · batch ×{batch_n}"
            self._clear_progress(
                busy=False,
                fast_warming=False,
                message=msg,
                error="",
            )
        except Exception as exc:
            failed_batch = int(getattr(self.engine, "stream_batch_size", 1) or 1)
            if failed_batch > 1 and _is_cuda_oom(exc):
                print(f"[batch] ×{failed_batch} warmup OOM — falling back to batch 1")
                self._note_batch_oom(failed_batch)
                self._recover_after_oom(1)
                try:
                    _warm(1)
                    self._fast_warmed = True
                    self._clear_progress(
                        busy=False,
                        fast_warming=False,
                        message=f"Ready — batch ×{failed_batch} skipped (GPU full)",
                        error="",
                    )
                    return
                except Exception as exc2:
                    exc = exc2
            self._fast_warmed = False
            self._clear_progress(
                busy=False,
                fast_warming=False,
                error=str(exc),
                message="Speed boost unavailable — running at normal speed",
            )
            print(f"[warmup] failed: {exc}")

    def _batch_is_auto(self) -> bool:
        return _clip_batch(self.status().get("batch")) <= 0

    def _hw_key(self) -> str:
        """Profile key: this GPU, model, steps and speed boost."""
        from .hw_profile import profile_key

        engine = self.engine
        gpu = ""
        try:
            device = getattr(engine, "device", None)
            if getattr(device, "type", "") == "cuda":
                import torch

                gpu = str(torch.cuda.get_device_name(device.index or 0))
        except Exception:
            gpu = ""
        st = self.status()
        steps = int(st.get("steps") or STREAM_DEFAULT_STEPS)
        boost = bool(st.get("fast_mode")) and bool(getattr(engine, "compile_model", False))
        return profile_key(gpu, checkpoint_label(engine.checkpoint), steps, boost)

    def _auto_batch_plan(self) -> int:
        """Batch Auto would stream at, from what this PC measured before."""
        from .hw_profile import load_failed, load_rates, plan_batch

        used = gpu_used_fraction(self.engine)
        if used is not None and not should_auto_batch2(used):
            self._set_status(message=f"GPU at {used:.0%} VRAM, keeping batch 1")
            return 1
        key = self._hw_key()
        self._batch_rates = load_rates(key)
        batch, _ = plan_batch(
            self._batch_rates, self._gen_cap(), STREAM_BATCH_MAX, failed=load_failed(key)
        )
        return batch

    def _tune_batch(self, warm: Callable[[int, str], None]) -> None:
        """Batch Auto: time this PC, then settle on the size ``plan_batch`` picks.

        Timings are saved per GPU/model, so this costs one extra warmup the
        first time and nothing after. A fast card that reaches the target at
        ×1 is never tried bigger; a size that runs out of memory is remembered.
        """
        from .hw_profile import TIME_RUNS, load_failed, load_rates, plan_batch, save_rate

        used = gpu_used_fraction(self.engine)
        if used is not None and not should_auto_batch2(used):
            return
        key = self._hw_key()
        rates = load_rates(key)
        failed = load_failed(key)
        target = self._gen_cap()
        engine = self.engine
        for _ in range(4):
            cur = int(getattr(engine, "stream_batch_size", 1) or 1)
            if cur not in rates:
                self._set_status(message=f"Timing batch ×{cur} on this PC")
                rates = save_rate(key, cur, engine.time_batch(cur, TIME_RUNS))
                self._batch_rates = rates
            batch, measure = plan_batch(rates, target, STREAM_BATCH_MAX, failed=failed)
            nxt = measure or batch
            if nxt == cur:
                break
            engine.set_stream_batch_size(nxt)
            try:
                warm(nxt, "Tuning for this PC")
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                print(f"[batch] ×{nxt} ran out of memory while tuning")
                self._note_batch_oom(nxt)
                failed = load_failed(key)
                self._recover_after_oom(cur)
                warm(cur, "Preparing stream")
        self._batch_rates = rates
        n = int(getattr(engine, "stream_batch_size", 1) or 1)
        secs = rates.get(n)
        if secs:
            print(f"[batch] auto ×{n}: {n / secs:.1f} keys/s here, target {target:.1f}")

    def _note_batch_oom(self, batch: int) -> None:
        from .hw_profile import save_failed

        try:
            save_failed(self._hw_key(), batch)
        except Exception:
            pass

    def _recover_after_oom(self, batch: int) -> None:
        try:
            self.engine.set_stream_batch_size(batch)
            self.engine._restore_eager_model()
            self.engine._compile_failed = False
        except Exception:
            pass
        if self.engine.device.type == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def _note_live_call(self, batch: int, seconds: float) -> None:
        if seconds > 0.0:
            self._live_calls.append((int(batch), float(seconds)))
            if len(self._live_calls) > 2000:
                del self._live_calls[:1000]

    def _save_live_rate(self) -> None:
        """Fold this stream's real call times into the PC profile, so Auto
        follows what the card manages with a game or OBS running too."""
        from .hw_profile import save_rate

        calls, self._live_calls = self._live_calls, []
        if not calls:
            return
        batch = calls[-1][0]
        secs = sorted(s for b, s in calls if b == batch)
        if len(secs) < 20:
            return
        try:
            self._batch_rates = save_rate(
                self._hw_key(), batch, secs[len(secs) // 2], weight=0.3
            )
        except Exception as exc:
            print(f"[batch] could not save timing: {exc}")

    def _pick_batch(self) -> None:
        """Set the stream batch size *before* torch.compile so we capture once.

        A fixed Batch setting is used as-is. Auto starts at what this PC's
        timings say; warmup then tunes it (see ``_tune_batch``).
        """
        if self._batch_picked or self._streaming:
            return
        self._batch_picked = True
        want = _clip_batch(self.status().get("batch"))
        if want <= 0:
            want = self._auto_batch_plan()
        try:
            self.engine.set_stream_batch_size(want)
        except Exception as exc:
            print(f"[batch] could not set ×{want}: {exc}")
            return
        if want > 1:
            self._set_status(message=f"Compiling batch ×{want}")

    def _ensure_compile_ready(self) -> None:
        """Block Generate / Stream until Fast compile+warmup finishes."""
        if not bool(self.status().get("fast_mode")):
            return
        if self.status().get("fast_warming"):
            raise RuntimeError("The stream is still getting ready. Try again in a moment.")
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
            "pose_cfg",
            "id_cfg",
            "frame_blend",
            "inbetweens",
            "interpolate",
            "max_fps",
            "hold_last",
            "track_fps",
            "drive_pose",
            "show_mesh",
            "show_hair",
            "show_outline",
            "show_brows",
            "show_eyes",
            "show_nose",
            "show_mouth",
            "show_iris_overlay",
            "show_skeleton",
            "show_limiters",
            "mirror",
            "use_iris",
            "use_body",
            "fast_mode",
            "compile_model",
            "batch",
            "batch2",
            "auto_sync_track",
            "camera_index",
            "travel_box",
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
        if "pose_cfg" in updates:
            try:
                updates["pose_cfg"] = _clip_cfg(float(updates["pose_cfg"]))
            except (TypeError, ValueError):
                updates.pop("pose_cfg", None)
        if "id_cfg" in updates:
            try:
                updates["id_cfg"] = _clip_cfg(float(updates["id_cfg"]))
            except (TypeError, ValueError):
                updates.pop("id_cfg", None)
        if "frame_blend" in updates:
            try:
                updates["frame_blend"] = _clip_blend(float(updates["frame_blend"]))
            except (TypeError, ValueError):
                updates.pop("frame_blend", None)
        if "hold_last" in updates:
            updates["hold_last"] = bool(updates["hold_last"])
        if "interpolate" in updates:
            updates["interpolate"] = interpolate_on(updates["interpolate"])
        if "inbetweens" in updates:
            updates["inbetweens"] = _clip_inbetweens(updates["inbetweens"])
        if "max_fps" in updates:
            updates["max_fps"] = _clip_max_fps(updates["max_fps"])
        if "batch2" in updates:
            # Old Batch ×2 switch (dev panel / older desks).
            updates["batch"] = 2 if updates.pop("batch2") else 1
        if "batch" in updates:
            updates["batch"] = _clip_batch(updates["batch"])
        old_travel = None
        if "travel_box" in updates:
            old_travel = self._status.get("travel_box")
            updates["travel_box"] = normalize_travel_box(
                {**normalize_travel_box(old_travel), **(updates["travel_box"] or {})}
                if isinstance(updates["travel_box"], dict)
                else updates["travel_box"]
            )
        if "show_limiters" in updates:
            updates["show_limiters"] = bool(updates["show_limiters"])
        overlay_touched = any(
            k in updates
            for k in ("show_mesh", "show_hair", "show_limiters", "travel_box", *OVERLAY_PART_KEYS)
        )
        with self._lock:
            self._status.update(updates)
            if overlay_touched:
                any_mesh = any(
                    bool(self._status.get(k, True)) for k in OVERLAY_PART_KEYS
                )
                if "show_mesh" in updates:
                    on = bool(updates["show_mesh"])
                    for key in OVERLAY_PART_KEYS:
                        if key not in updates:
                            self._status[key] = on
                    any_mesh = any(
                        bool(self._status.get(k, True)) for k in OVERLAY_PART_KEYS
                    )
                self._status["show_mesh"] = any_mesh
            snap = dict(self._status)
        if "camera_index" in updates:
            try:
                self._remember_camera(int(updates["camera_index"]))
            except (TypeError, ValueError):
                pass
        if "steps" in updates:
            try:
                self.engine.num_steps = int(updates["steps"])
            except Exception:
                pass
        if "pose_cfg" in updates or "id_cfg" in updates:
            try:
                self.engine.set_guidance(
                    pose_cfg=updates.get("pose_cfg"),
                    id_cfg=updates.get("id_cfg"),
                )
            except Exception:
                pass
        if any(
            key in updates
            for key in (
                "steps",
                "pose_cfg",
                "id_cfg",
                "frame_blend",
                "inbetweens",
                "interpolate",
                "max_fps",
                "hold_last",
            )
        ):
            from .ui_session import save_ui_session

            save_ui_session(
                steps=self._status.get("steps"),
                pose_cfg=self._status.get("pose_cfg"),
                id_cfg=self._status.get("id_cfg"),
                frame_blend=self._status.get("frame_blend"),
                inbetweens=self._status.get("inbetweens"),
                interpolate=self._status.get("interpolate"),
                max_fps=self._status.get("max_fps"),
                hold_last=self._status.get("hold_last"),
            )
        if "hold_last" in updates:
            self.engine.set_hold_last(bool(updates["hold_last"]))
        if "mirror" in updates:
            want = bool(updates["mirror"])
            self.tracker.mirror = want
            if self._lab_drive or getattr(self, "_lab_seen_online", False):
                try:
                    self._lab_ack("set_mirror", {"on": want})
                except Exception as exc:
                    print(f"[lab-harness] set_mirror failed: {exc}", flush=True)
        if "travel_box" in updates:
            from .ui_session import save_ui_session

            save_ui_session(travel_box=updates["travel_box"])
            self._save_character_limiters(updates["travel_box"])
            # Clear the stopped-tracking preview first so a live emit shows the
            # camera pose at the new cap, not the slider's extreme.
            self._refresh_travel_preview(old_travel, updates["travel_box"])
            # Explicit UI / session edit — push full box; do not push feel caps.
            # Applies while tracking is on.
            self._push_lab_limiters(user_edit=True)
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
        if "compile_model" in updates:
            want = bool(updates["compile_model"])
            try:
                self.engine.set_compile_model(want)
                from .ui_session import save_ui_session

                save_ui_session(compile_model=want)
                self._status["compile_model"] = want
                # Queue warmup until the DiT + reference are actually loaded.
                if (
                    want
                    and bool(self.status().get("fast_mode"))
                    and not self._streaming
                    and getattr(self.engine, "_ready", False)
                    and getattr(self.engine, "_ref_latent", None) is not None
                ):
                    self._fast_warmed = False
                    self._run_fast_warmup_if_needed(force=True)
                snap = self.status()
                snap["compile_model"] = want
            except Exception as exc:
                self._set_status(error=str(exc), compile_model=want)
                snap = self.status()
                snap["compile_model"] = want
        if any(k in updates for k in ("max_fps", "inbetweens", "interpolate")) and not self._streaming:
            # Auto batch aims at the key rate these set; re-plan at next warmup.
            if self._batch_is_auto():
                self._batch_picked = False
                self._fast_warmed = False
        if "batch" in updates:
            from .ui_session import save_ui_session

            save_ui_session(batch=updates["batch"])
            # Every size is its own compiled graph: pick it again and re-warm.
            # Mid-stream the running size stays until the next Start.
            self._batch_picked = False
            self._fast_warmed = False
            if not self._streaming:
                try:
                    self._pick_batch()
                    # Keep the compiled wrapper. Warmup captures the new batch
                    # shape as another inductor graph (no full recompile).
                    if bool(self.status().get("fast_mode")) and getattr(
                        self.engine, "_model_compiled", False
                    ):
                        self._run_fast_warmup_if_needed(force=True)
                except Exception as exc:
                    self._set_status(error=str(exc))
                snap = self.status()
        self._emit({"type": "status", "status": snap})
        if overlay_touched and self._last_image is not None:
            self._emit(
                {
                    "type": "frame",
                    **self._frame_payload(self._last_image, self._last_overlay_kps),
                }
            )
        return snap

    def list_cameras(self) -> list[dict[str, Any]]:
        from .lab_harness import lab as lab_harness

        try:
            packet = lab_harness.status(merge_frame=False)
            raw = packet.get("cameras") if isinstance(packet, dict) else None
            if packet.get("online") and isinstance(raw, list) and raw:
                cameras: list[dict[str, Any]] = []
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    try:
                        idx = int(item.get("index", 0))
                    except (TypeError, ValueError):
                        continue
                    cameras.append(
                        {
                            "index": idx,
                            "name": str(item.get("name") or f"Camera {idx}"),
                        }
                    )
                if cameras:
                    return cameras
        except Exception:
            pass
        cams = list_cameras()
        return [{"index": i, "name": name} for i, name in cams]

    def preferred_camera(self) -> int:
        """Last camera the user chose, if it is still connected."""
        try:
            cams = self.list_cameras()
            saved_index, saved_name = _remembered_camera()
            if saved_name or saved_index is not None:
                from track_lab.backend.cameras import pick_default

                return int(pick_default(cams, saved_index, saved_name=saved_name))
            pairs = [(int(c["index"]), str(c.get("name") or "")) for c in cams]
            return int(pick_default_camera_index(pairs))
        except Exception:
            try:
                return int(self._status.get("camera_index") or 0)
            except (TypeError, ValueError):
                return 0

    def _remember_camera(self, index: int) -> None:
        from track_lab.backend.cameras import save_camera_index

        name = ""
        try:
            for cam in self.list_cameras():
                if int(cam.get("index", -1)) == int(index):
                    name = str(cam.get("name") or "")
                    break
        except Exception:
            name = ""
        save_camera_index(int(index), name)

    def _reset_live_origin(self, *, reason: str = "") -> None:
        self._live_origin_keypoints = None
        self._live_origin_crop_key = None
        self._live_origin_skel_kind = "none"
        self._prev_controls = None
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
        if self._last_image is None and self._character_path() is None:
            return
        cam = int(self.preferred_camera())
        self._reset_live_origin(reason="start_tracking")
        self._capture_session_seen = 0
        self._set_track_status(
            busy=True,
            track_busy=True,
            message="Starting tracking…",
            track_message="Starting…",
            error="",
        )
        try:
            probe = self._wait_lab_for_tracking()
            self._start_lab_tracking(cam, probe)
        except Exception as exc:
            self._tracking = False
            self._lab_drive = False
            self._set_track_status(
                busy=False,
                track_busy=False,
                tracking=False,
                error=str(exc),
                message="Tracking failed",
                track_message="Tracking failed",
            )
            raise

    def _wait_lab_for_tracking(self, timeout: float = 90.0) -> dict[str, Any]:
        """Hook the host splash already started. Only wait if it is still warming."""
        from .lab_process import BOOT_WAIT, connect_lab, harness_ok
        from .lab_harness import lab as lab_harness, tracker_loaded, wait_loaded

        def _lab_wait(frac: float) -> None:
            self._set_track_status(
                busy=True,
                message="Waiting for Track Lab…",
                track_message="Waiting for Track Lab…",
                progress=max(0.0, min(1.0, float(frac))),
                progress_label="Connecting Track Lab",
                progress_kind="lab",
            )

        wait = max(float(timeout), float(BOOT_WAIT))
        ok = harness_ok()
        packet = lab_harness.status(merge_frame=False)
        _runtime_log(
            f"track hook harness_ok={ok} online={bool(packet.get('online'))} "
            f"loaded={packet.get('loaded')} error={packet.get('error') or '—'}"
        )
        if packet.get("online"):
            packet["handshake"] = True
        else:
            packet = connect_lab(timeout=wait, on_wait=_lab_wait)
            _runtime_log(
                f"track connect online={bool(packet.get('online'))} "
                f"loaded={packet.get('loaded')} error={packet.get('error') or '—'}"
            )
        if not packet.get("online"):
            raise RuntimeError(
                str(packet.get("error") or "Track Lab is still starting — wait a moment and try again")
            )
        if tracker_loaded(packet):
            return packet
        self._set_track_status(
            busy=True,
            message="Loading Track Lab tracker…",
            track_message="Loading tracker…",
            progress_label="Loading tracker",
            progress_kind="lab",
        )

        def _load_wait(frac: float) -> None:
            self._set_track_status(
                busy=True,
                message="Loading Track Lab tracker…",
                track_message="Loading tracker…",
                progress=max(0.0, min(1.0, float(frac))),
                progress_label="Loading tracker",
                progress_kind="lab",
            )

        last = wait_loaded(timeout=min(12.0, wait), on_wait=_load_wait)
        _runtime_log(
            f"track loaded={bool(last.get('loaded'))} online={bool(last.get('online'))} "
            f"error={last.get('error') or '—'}"
        )
        if last.get("online"):
            return last
        raise RuntimeError(
            str(last.get("error") or "Track Lab tracker is still loading — wait a moment and try again")
        )

    def _start_lab_tracking(self, cam: int, probe: dict[str, Any]) -> None:
        source = "ifm" if str(probe.get("source") or "") == "ifm" else "camera"
        self._set_track_status(busy=True, message="Fitting rest mesh…", track_message="Fitting rest…")
        self._sync_lab_character()
        self._restore_character_fit()
        # Lab wins on connect — take its travel box before start, never push desk.
        self.adopt_lab_travel_box(probe)
        body: dict[str, Any] = {"source": source}
        if source == "camera":
            body["camera"] = cam
            body["mirror"] = bool(self._status.get("mirror"))
        self._set_track_status(busy=True, message="Starting Track Lab…", track_message="Starting…")
        ack = self._lab_ack("start", body)
        self._tracking = True
        self._lab_drive = True
        self._travel_preview_kps = None
        self._adopt_lab_hair()
        # Lab wins on connect / start — adopt its box; do not overwrite the lab.
        packet = self._lab_packet_from_ack(ack) or {}
        nested = ack.get("status") if isinstance(ack.get("status"), dict) else {}
        self.adopt_lab_travel_box(nested or packet)
        self._set_track_status(
            busy=False,
            track_busy=False,
            tracking=True,
            message="Tracking on",
            track_message="Tracking on",
            error="",
        )

    def _start_legacy_tracking(self, cam: int, st: dict[str, Any]) -> None:
        self.tracker.mirror = bool(st.get("mirror"))
        self.tracker.set_mouth_osf_map(None)
        use_iris = bool(st.get("use_iris", True))
        use_body = bool(st.get("use_body", True))
        if not DEVELOPER:
            use_iris = True
            use_body = True
        self.tracker.use_iris = use_iris
        self.tracker.use_skeleton = use_body
        self.tracker.start(cam, camera_name=f"Camera {cam}")
        self._tracking = True
        self._lab_drive = False
        self._travel_preview_kps = None
        self._last_lab_hair = None
        self._lab_rest_hair = None
        self._set_status(
            busy=False,
            tracking=True,
            message="Tracking on",
            track_message="Tracking on — auto-centering…",
        )

    def _write_lab_source(self) -> Path | None:
        dest = package_root() / "track_lab" / "input" / "source.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self._last_image is not None:
            self._last_image.convert("RGB").save(dest, format="PNG")
            return dest
        ref = self._ref_path
        if ref is not None and ref.is_file():
            if ref.suffix.lower() == ".vtm":
                from .character_pack import read_character_preview_png

                dest.write_bytes(read_character_preview_png(ref))
                return dest
            return ref
        return None

    def _push_lab_limiters(self, *, user_edit: bool = False) -> None:
        """Send the full travel box to Track Lab. Only for explicit desk edits."""
        if not user_edit:
            return
        from .lab_harness import lab as lab_harness

        box = normalize_travel_box(self._status.get("travel_box"))
        try:
            ack = lab_harness.command("set_travel", dict(box))
        except Exception as exc:
            print(f"[lab-harness] set_travel failed: {exc}", flush=True)
            return
        if not isinstance(ack, dict):
            return
        if ack.get("ok") is False or not ack.get("online", True):
            err = str(ack.get("error") or "Track Lab is not running")
            print(f"[lab-harness] set_travel failed: {err}", flush=True)
            return
        # Desk owns the sliders from here. A camera frame packed before this
        # edit must not snap Look up back.
        self._travel_from_desk = True
        # Lab may re-normalize; adopt ack box without echoing another set_travel.
        nested = ack.get("status") if isinstance(ack.get("status"), dict) else {}
        raw = nested.get("travel_box") if isinstance(nested, dict) else None
        if not isinstance(raw, dict):
            raw = ack.get("travel_box")
        if isinstance(raw, dict):
            self.adopt_lab_travel_box({"travel_box": raw}, user_edit=True)
        self._emit_live_limiter_pose(ack)

    def _emit_live_limiter_pose(self, ack: dict[str, Any] | None) -> None:
        """Show the re-capped pose now. Tracking stays on."""
        if not getattr(self, "_tracking", False):
            return
        if getattr(self, "_pose_frozen", False) or getattr(self, "_mesh_edited", False):
            return
        frame = ack.get("frame") if isinstance(ack, dict) and isinstance(ack.get("frame"), dict) else None
        driven = None
        if isinstance(frame, dict) and ("keypoints" in frame or "generation" in frame):
            try:
                driven = self._lab_overlay_keypoints(frame)
            except Exception:
                driven = None
        if driven is None:
            try:
                driven = self._lab_overlay_keypoints()
            except Exception:
                return
        image = self._last_image
        if driven is not None and image is not None:
            self._emit({"type": "frame", **self._frame_payload(image, driven)})

    def adopt_lab_travel_box(self, packet: dict[str, Any] | None, *, user_edit: bool = False) -> bool:
        """Copy Track Lab's travel_box into desk status. Never echoes set_travel."""
        if not isinstance(packet, dict):
            return False
        if getattr(self, "_travel_from_desk", False) and not user_edit:
            return False
        raw = packet.get("travel_box")
        if not isinstance(raw, dict):
            return False
        incoming = normalize_travel_box(raw)
        current = normalize_travel_box(self._status.get("travel_box"))
        if incoming == current:
            return False
        from .ui_session import save_ui_session

        with self._lock:
            self._status["travel_box"] = incoming
            snap = dict(self._status)
        save_ui_session(travel_box=incoming)
        if user_edit:
            self._save_character_limiters(incoming)
        self._emit({"type": "status", "status": snap})
        return True

    def _save_character_limiters(self, box: Any) -> None:
        """Limiter edits belong to the loaded character's ``.vtm``."""
        from .character_fit import update_character_fit

        ident = str(self._status.get("character_id") or "")
        if not ident:
            return
        try:
            update_character_fit(ident, {"travel_box": normalize_travel_box(box)})
        except Exception as exc:
            print(f"Limiters did not save into the character: {exc}")

    def _apply_character_limiters(self) -> None:
        """Install the limiters saved in the loaded character's ``.vtm``.

        Packs made before limiters were saved per character adopt the desk's
        current box, so each character keeps its own from then on.
        """
        from .character_fit import read_character_fit

        ident = str(self._status.get("character_id") or "")
        if not ident:
            return
        try:
            saved = read_character_fit(ident).get("travel_box")
        except Exception as exc:
            print(f"Character limiters unreadable: {exc}")
            return
        if not isinstance(saved, dict):
            self._save_character_limiters(self._status.get("travel_box"))
            return
        from .ui_session import save_ui_session

        box = normalize_travel_box(saved)
        old = self._status.get("travel_box")
        with self._lock:
            self._status["travel_box"] = box
            snap = dict(self._status)
        save_ui_session(travel_box=box)
        # The character owns the limiters; Track Lab's stale copy must not win.
        self._travel_from_desk = True
        if normalize_travel_box(old) != box:
            self._emit({"type": "status", "status": snap})
        if getattr(self, "_lab_seen_online", False):
            self._push_lab_limiters(user_edit=True)

    def _lab_ack(self, op: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        from .lab_harness import lab as lab_harness

        ack = lab_harness.command(op, body)
        nested = ack.get("status") if isinstance(ack.get("status"), dict) else {}
        err = str(ack.get("error") or nested.get("error") or "")
        if ack.get("ok") is False or (err and not ack.get("ok", True)):
            raise RuntimeError(err or f"Track Lab {op} failed")
        return ack

    def _limiter_rest(self) -> np.ndarray | None:
        """The character's rest pose. Limiter walls are boxed around this, never the live pose."""
        rest = getattr(getattr(self, "engine", None), "_ref_keypoints", None)
        if rest is None:
            rest = getattr(self, "_last_overlay_kps", None)
        return None if rest is None else np.asarray(rest, dtype=np.float32)

    def _refresh_travel_preview(self, old_box: Any, new_box: Any) -> None:
        """Walk the overlay to the slider that just moved so the max is visible."""
        if getattr(self, "_tracking", False):
            self._travel_preview_kps = None
            return
        axis = changed_preview_axis(old_box, new_box)
        rest = self._limiter_rest()
        if axis is None or rest is None:
            self._travel_preview_kps = None
            return
        self._travel_preview_kps = preview_travel_pose(rest, new_box, axis)

    def _clamp_travel_box(
        self,
        driven: np.ndarray,
        hair: list | None = None,
    ) -> tuple[np.ndarray, list | None]:
        out, hair_out, _scale = apply_travel_box(
            driven,
            getattr(self.engine, "_ref_keypoints", None),
            self._status.get("travel_box"),
            hair=hair,
        )
        return out, hair_out

    def _same_lab_still(self) -> bool:
        return stills_match(
            self._last_image,
            package_root() / "track_lab" / "input" / "source.png",
        )

    def _lab_generation(self, packet: dict[str, Any] | None) -> int:
        from .lab_harness import lab_packet_generation

        return lab_packet_generation(packet)

    def _lab_packet_from_ack(self, ack: dict[str, Any] | None) -> dict[str, Any] | None:
        from .lab_harness import lab_packet_from_ack

        return lab_packet_from_ack(ack)

    def _note_lab_generation(self, packet: dict[str, Any] | None) -> None:
        gen = self._lab_generation(packet)
        if gen:
            seen = int(getattr(self, "_lab_seen_generation", 0) or 0)
            self._lab_seen_generation = max(seen, gen)

    def _accept_lab_overlay(self, packet: dict[str, Any] | None) -> None:
        if not isinstance(packet, dict):
            return
        # None means this still is not in Track Lab yet. Generation 0 is a
        # real still — dropping those frames left the cel on the rest pose
        # while blink / look meters kept updating.
        if "generation" not in packet and "keypoints" not in packet:
            return
        gen = self._lab_generation(packet)
        self._lab_overlay_gen = gen
        self._note_lab_generation(packet)

    def _lab_overlay_current(self, packet: dict[str, Any] | None) -> bool:
        want = getattr(self, "_lab_overlay_gen", 0)
        if want is None:
            return False
        gen = self._lab_generation(packet)
        if not gen:
            return int(want) <= 0
        return gen >= int(want)

    def adopt_lab_overlay(
        self,
        packet: dict[str, Any] | None = None,
        *,
        emit: bool = False,
    ) -> bool:
        """Copy Track Lab overlay onto the desk cel. Authoring is not live-only."""
        with self._lock:
            frozen = bool(self._pose_frozen or self._mesh_edited)
        if frozen:
            return False
        driven = self._lab_overlay_keypoints(packet)
        if driven is None:
            return False
        image = self._last_image
        if emit and image is not None:
            self._emit({"type": "frame", **self._frame_payload(image, driven)})
        return True

    def _lab_overlay_keypoints(self, frame: dict[str, Any] | None = None) -> np.ndarray | None:
        from .lab_harness import (
            frame_image_wh,
            hair_from_frame,
            lab as lab_harness,
            overlay_from_frame,
        )

        if frame is None or "keypoints" not in frame:
            frame = lab_harness.frame()
        if not isinstance(frame, dict):
            if not getattr(self, "_lab_drive", False):
                self._lab_seen_online = False
            return None
        self._note_lab_generation(frame)
        if not self._lab_overlay_current(frame):
            return None
        self._lab_seen_online = True
        lab_wh = frame_image_wh(frame)
        if lab_wh is not None:
            self._lab_image_wh = lab_wh
        image = self._last_image
        w, h = (image.size if image is not None else (0, 0))
        hair = hair_from_frame(frame, width=w, height=h)
        driven = overlay_from_frame(frame, width=w, height=h)
        if driven is None:
            if hair is not None:
                self._last_lab_hair = hair
                if self._lab_rest_hair is None and hair:
                    self._lab_rest_hair = [dict(seg) for seg in hair]
            self.adopt_lab_travel_box(frame)
            return None
        # Lab already limited walk / look / rotate. Do not re-clamp here —
        # a second apply_walk_box squashes the look the lab authored.
        self.adopt_lab_travel_box(frame)
        driven = self._apply_drag_to_overlay(driven)
        if hair is not None:
            self._last_lab_hair = hair
            if self._lab_rest_hair is None and hair:
                self._lab_rest_hair = [dict(seg) for seg in hair]
        self._last_overlay_kps = driven.copy()
        self._driven_keypoints = driven.copy()
        self._last_good_keypoints = driven.copy()
        self._body_lost = False
        return driven

    def _apply_drag_to_overlay(self, driven: np.ndarray) -> np.ndarray:
        """Keep the grabbed point on the pointer while the rest of the mesh tracks."""
        with self._lock:
            slots = set(self._drag_slots)
            xy = self._drag_xy
            image = self._last_image
        if not slots or xy is None or image is None:
            return driven
        w, h = image.size
        nx, ny = pixels_to_normalized(float(xy[0]), float(xy[1]), w, h)
        out = np.asarray(driven, dtype=np.float32).copy()
        for j in slots:
            if 0 <= int(j) < NUM_KEYPOINTS:
                out[int(j), 0] = nx
                out[int(j), 1] = ny
                out[int(j), 2] = max(float(out[int(j), 2]), 0.85)
                out[int(j), 3] = 1.0
        return out

    def _desk_px_to_lab(self, x: float, y: float) -> tuple[float, float]:
        """Desk jpeg pixels → Track Lab character_px (lab image_wh)."""
        image = getattr(self, "_last_image", None)
        lab = getattr(self, "_lab_image_wh", None)
        if image is None or not lab:
            return float(x), float(y)
        dw, dh = image.size
        lw, lh = int(lab[0]), int(lab[1])
        if dw <= 0 or dh <= 0 or lw <= 0 or lh <= 0 or (dw == lw and dh == lh):
            return float(x), float(y)
        return float(x) * float(lw) / float(dw), float(y) * float(lh) / float(dh)

    def _nudge_lab_point(self, idx: int, x: float, y: float) -> None:
        try:
            px, py = self._desk_px_to_lab(x, y)
            self._lab_ack("set_point", {"id": int(idx), "x": float(px), "y": float(py)})
        except Exception as exc:
            print(f"lab set_point failed: {exc}")

    def stop_tracking(self) -> None:
        from .lab_harness import lab as lab_harness

        try:
            lab_harness.command("stop")
        except Exception:
            pass
        if self._tracking and not self._lab_drive:
            try:
                self.tracker.stop(settle=True)
            except Exception:
                pass
        self._tracking = False
        self._lab_drive = False
        self._reset_live_origin(reason="stop_tracking")
        self._set_track_status(
            tracking=False,
            track_busy=False,
            busy=False,
            message="Tracking off",
            track_message="Tracking off",
            body_label="",
        )

    def calibrate_ref(self) -> None:
        if self._character_path() is None:
            return
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

    def apply_lab_calibrate(self, ack: dict[str, Any] | None = None) -> None:
        """Let harness rest land on the cel. Freeze would keep the broken pose."""
        with self._lock:
            self._pose_frozen = False
            self._mesh_edited = False
        nested = ack.get("status") if isinstance(ack, dict) and isinstance(ack.get("status"), dict) else {}
        packet = nested if nested.get("keypoints") else None
        self.adopt_lab_overlay(packet, emit=True)
        calib = nested.get("calib") if isinstance(nested.get("calib"), dict) else {}
        capturing = str(calib.get("capturing") or "")
        err = str((ack or {}).get("error") or nested.get("error") or "")
        if capturing == "rest":
            message = "Calibrating rest — hold still"
        elif err:
            message = err
        else:
            message = "Rest captured"
        self._set_status(
            pose_frozen=False,
            message=message,
            track_message=message,
            error=err,
        )

    def calibrate_lab_rest(self, name: str = "rest") -> dict[str, Any]:
        ack = self._lab_ack("calibrate", {"id": str(name or "rest")})
        self.apply_lab_calibrate(ack)
        return ack

    def recenter(self) -> None:
        if self._lab_drive or getattr(self, "_lab_seen_online", False):
            self.calibrate_lab_rest("rest")
            return
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
        with self._lock:
            frozen = bool(self._pose_frozen or self._mesh_edited)
            if frozen and self._driven_keypoints is not None:
                return np.asarray(self._driven_keypoints, dtype=np.float32).copy()
        st = self.status()
        drive = bool(st.get("drive_pose", True))
        if not DEVELOPER:
            drive = True
        if drive and (self._lab_drive or getattr(self, "_lab_seen_online", False)):
            driven = self._lab_overlay_keypoints()
            if driven is not None:
                return driven
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

    def _pixel_hair_to_norm(self, segments: list, width: int, height: int) -> list:
        w = max(float(width), 1.0)
        h = max(float(height), 1.0)
        out = []
        for seg in segments:
            cls = str(seg.get("class") or "")
            poly = seg.get("polygon") or []
            if not poly:
                continue
            pts = []
            for x, y in poly:
                pts.append([float(x) / w * 2.0 - 1.0, float(y) / h * 2.0 - 1.0])
            rec = {"class": cls, "polygon": pts}
            out.append(rec)
        return out

    def _current_hair_maps(
        self,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        segs = self._followed_hair_segments(keypoints)
        if segs:
            try:
                return rasterize_hair_maps(segs)
            except Exception:
                pass
        return empty_hair_maps()

    def _rest_hair_segments(self) -> list | None:
        if self._lab_rest_hair:
            return [dict(seg) for seg in self._lab_rest_hair]
        if self._last_lab_hair and self._lab_drive:
            return [dict(seg) for seg in self._last_lab_hair]
        if self._hair_rig is None and not self._hair_capture_done:
            self._maybe_capture_hair()
        if self._hair_rig is None:
            return None
        try:
            segs = rest_hair(self._hair_rig)
        except Exception:
            return None
        return segs or None

    def _followed_hair_segments(
        self, keypoints: np.ndarray | None = None
    ) -> list | None:
        """Three hair parts in norm_crop.

        Track Lab already yaws / follows hair in character pixels. When a
        harness mesh is present, copy that instead of rebuilding it here.
        """
        if self._last_lab_hair is not None:
            return self._last_lab_hair or None
        if self._lab_drive:
            return None
        if self._hair_rig is None and not self._hair_capture_done:
            self._maybe_capture_hair()
        if self._hair_rig is None:
            return None
        kps = keypoints if keypoints is not None else self._last_overlay_kps
        if kps is None:
            return None
        try:
            segs = follow_hair(self._hair_rig, np.asarray(kps, dtype=np.float32))
        except Exception:
            return None
        return segs or None

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
        # Uncalibrated RelativePose yaw/pitch/roll are absolute euler angles,
        # not deltas from Center. Passing them as "calibrated" relative angles
        # warps rest pose and then swallows real nods (geom delta ≈ 0 at
        # origin lock, so the 15° resting pitch looks like the character).
        # Landmark geom fallback in extract_controls handles the wait.
        rel_calibrated = rel is not None and bool(getattr(rel, "calibrated", False))
        head_yaw = float(rel.yaw) if rel_calibrated else None
        head_pitch = float(rel.pitch) if rel_calibrated else None
        head_roll = float(rel.roll) if rel_calibrated else None
        head_tx = float(rel.nx) if rel_calibrated else None
        head_ty = float(rel.ny) if rel_calibrated else None
        held: list = []

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
                reference_rig=getattr(self.engine, "_ref_rig", None),
                prev_controls=self._prev_controls,
                controls_out=held,
                motion=motion_caps(self._status.get("travel_box")),
            )
        except Exception as exc:
            print(f"[pose-diag] retarget failed: {exc}")
            return self._driven_keypoints

        try:
            self.engine._last_driven_body = np.asarray(driven[30:37], dtype=np.float32).copy()
        except Exception:
            pass

        driven = np.asarray(driven, dtype=np.float32)
        if held:
            self._prev_controls = held[-1]
        ctrl = held[-1] if held else self._prev_controls
        if self._pose_keys and ctrl is not None and not self._pose_frozen:
            driven = apply_keys(
                driven,
                params_from_controls(ctrl),
                self._pose_keys,
                live_roll_deg=float(getattr(ctrl, "head_roll_deg", 0.0) or 0.0),
            )
        driven, _hair = self._clamp_travel_box(driven)
        self._driven_keypoints = driven.copy()
        self._last_good_keypoints = driven.copy()
        self._last_overlay_kps = driven.copy()
        self._mouth_snapped = False
        return driven

    def _ensure_ref(self) -> bool:
        if getattr(self.engine, "_ref_latent", None) is not None:
            return True
        path = self._character_path()
        if path is None:
            return False
        self.apply_reference(path)
        return True

    def generate_once(self) -> None:
        if self._streaming:
            raise RuntimeError("Busy — stop the stream or wait")
        if self._frame_in_flight or self._gen_busy:
            raise RuntimeError("Busy — wait for the current generate")
        if self.status().get("fast_warming"):
            raise RuntimeError("The stream is still getting ready. Try again in a moment.")
        if not self._ensure_ref():
            return
        self.ensure_model(keep_bar=True)
        self._ensure_compile_ready()
        with self._lock:
            leftover = self._status.get("progress_kind") == "warmup"
        if leftover:
            self._clear_progress()
        kps = self._current_keypoints()
        if kps is None:
            return
        steps = int(self.status().get("steps") or STREAM_DEFAULT_STEPS)
        self._frame_in_flight = True
        self._set_status(busy=True, message="Generating…")
        current = np.asarray(kps, dtype=np.float32)
        batch_n = max(1, int(getattr(self.engine, "stream_batch_size", 1) or 1))
        # Same batch size the model was warmed at, or this one call recompiles.
        keypoints, hair_maps = pack_stream_batch(
            current,
            np.asarray(self._current_hair_maps(current), dtype=np.float32),
            None,
            None,
            batch_n,
        )
        self._enqueue_generate(
            steps=steps,
            streaming=False,
            keypoints=keypoints,
            hair_maps=hair_maps,
        )

    def _vcam_frame_size(self) -> tuple[int, int]:
        """Square DiT canvas — never a widescreen still or the desktop."""
        from .virtual_cam import vcam_even_size

        size = int(getattr(self.engine, "image_size", 768) or 768)
        image = self._last_image
        if image is not None:
            w, h = image.size
            if w == h and w >= 64:
                size = int(w)
        w, h = vcam_even_size(size, size)
        return w, h

    def _vcam_source(self) -> Image.Image | None:
        return self._last_image

    def start_virtual_cam(self) -> None:
        """Open the VTM Spark camera and keep the current still/gen picture pumping."""
        from .vcam_device import DEVICE_NAME
        from .virtual_cam import VCAM_FPS, get_virtual_cam

        w, h = self._vcam_frame_size()
        vcam = get_virtual_cam()
        try:
            device = vcam.start(w, h, fps=VCAM_FPS, source=self._vcam_source)
        except Exception as exc:
            self._vcam_wanted = False
            self._set_status(
                virtual_cam=False,
                virtual_cam_device="",
                virtual_cam_error=str(exc),
                virtual_cam_width=0,
                virtual_cam_height=0,
                error=str(exc),
                message="Virtual camera failed",
            )
            raise
        self._vcam_wanted = True
        seed = self._last_image
        if seed is None:
            seed = Image.new("RGB", (w, h), (8, 8, 8))
        vcam.send(seed)
        self._set_status(
            virtual_cam=True,
            virtual_cam_device=device or DEVICE_NAME,
            virtual_cam_error="",
            virtual_cam_width=w,
            virtual_cam_height=h,
            error="",
            message=f"Virtual camera on · {device or DEVICE_NAME} ({w}×{h})",
        )

    def stop_virtual_cam(self) -> None:
        from .virtual_cam import get_virtual_cam

        self._vcam_wanted = False
        get_virtual_cam().stop()
        self._set_status(
            virtual_cam=False,
            virtual_cam_device="",
            virtual_cam_error="",
            virtual_cam_width=0,
            virtual_cam_height=0,
            message="Virtual camera off",
        )

    def start_stream(self) -> None:
        if self._streaming:
            return
        if self.status().get("fast_warming"):
            raise RuntimeError("The stream is still getting ready. Try again in a moment.")
        if not self._ensure_ref():
            return
        self.ensure_model(keep_bar=True)
        self._ensure_compile_ready()
        self._streaming = True
        self._paused = False
        self._frame_in_flight = False
        self._last_gen_start = 0.0
        self._gen_hold_pending = False
        self._prev_stream_kps = None
        self._prev_stream_hair = None
        self._ema_frame = None
        self._ema_kps = None
        self._inbetween_prev = None
        self._inbetween_prev_kps = None
        self._reset_display_clock()
        self._display_busy = False
        self._drain_display_queue()
        self._set_status(
            streaming=True,
            paused=False,
            show_fps=0.0,
            gen_fps=0.0,
            busy=False,
        )
        # "Streaming" only once a picture is out; until then the bar stays.
        self._first_frame_pending = True
        self._set_progress(0.96, label="Starting stream", kind="warmup")
        self._schedule_next_frame()

    def _end_first_frame_wait(self, **status: Any) -> None:
        if not self._first_frame_pending:
            if status:
                self._set_status(**status)
            return
        self._first_frame_pending = False
        self._clear_progress(busy=False, **status)

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        self._streaming = False
        self._paused = False
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
        self._prev_stream_hair = None
        self._ema_frame = None
        self._ema_kps = None
        self._inbetween_prev = None
        self._inbetween_prev_kps = None
        self._reset_display_clock()
        self._drain_display_queue()
        self._save_live_rate()
        self._offload_pending = True
        self._end_first_frame_wait()
        self._set_status(
            streaming=False, paused=False, busy=False, message="Stream stopped"
        )
        self._maybe_offload_after_stop()

    def _offload_models(self) -> None:
        """Move DiT / VAE off the GPU after Stop stream. Tracking stays up."""
        if self._streaming or self._gen_busy or self.status().get("fast_warming"):
            self._offload_pending = True
            return
        try:
            moved = bool(self.engine.offload_to_cpu())
        except Exception as exc:
            print(f"[offload] {exc}")
            self._offload_pending = False
            return
        self._offload_pending = False
        if moved:
            self._fast_warmed = False
            self._set_status(
                models_on_gpu=False,
                message="Stream stopped — models offloaded",
            )

    def _maybe_offload_after_stop(self) -> None:
        if not self._offload_pending or self._streaming or self._gen_busy:
            return
        self._offload_models()

    def pause_stream(self) -> None:
        if not self._streaming or self._paused:
            return
        self._paused = True
        self._end_first_frame_wait()
        self._set_status(paused=True, message="Stream paused")

    def resume_stream(self) -> None:
        if not self._streaming:
            self.start_stream()
            return
        if not self._paused:
            return
        self._paused = False
        # The paused gap is not a key interval.
        self._last_call_done_t = 0.0
        self._set_status(paused=False, message="Streaming")
        if not self._frame_in_flight:
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        if not self._streaming or self._paused or self._frame_in_flight:
            return
        # Max FPS: hold the next DiT call so the GPU idles between keys.
        batch_n = max(1, int(getattr(self.engine, "stream_batch_size", 1) or 1))
        hold = gen_hold_s(
            self._gen_cap(),
            self._last_gen_start,
            time.perf_counter(),
            batch=batch_n,
        )
        if hold > 0.0:
            if not self._gen_hold_pending:
                self._gen_hold_pending = True
                threading.Timer(hold, self._resume_after_gen_hold).start()
            return
        self._frame_in_flight = True
        self._last_gen_start = time.perf_counter()
        kps = self._current_keypoints()
        if kps is None:
            self._frame_in_flight = False
            threading.Timer(0.05, self._schedule_next_frame).start()
            return
        current = np.asarray(kps, dtype=np.float32)
        hair = np.asarray(self._current_hair_maps(current), dtype=np.float32)
        batch_n = max(1, int(getattr(self.engine, "stream_batch_size", 1) or 1))
        keypoints, hair_maps = pack_stream_batch(
            current,
            hair,
            self._prev_stream_kps,
            self._prev_stream_hair,
            batch_n,
        )
        self._prev_stream_kps = current.copy()
        self._prev_stream_hair = hair.copy()
        steps = int(self.status().get("steps") or STREAM_DEFAULT_STEPS)
        self._enqueue_generate(
            steps=steps, streaming=True, keypoints=keypoints, hair_maps=hair_maps
        )

    def _gen_cap(self) -> float:
        """Keys/s the stream is held to: Max FPS, or Auto from the in-betweens."""
        return gen_cap(
            self._status.get("max_fps"),
            self._status.get("interpolate"),
            self._status.get("inbetweens"),
        )

    def _resume_after_gen_hold(self) -> None:
        self._gen_hold_pending = False
        self._schedule_next_frame()

    def _enqueue_generate(
        self,
        *,
        steps: int,
        streaming: bool,
        keypoints: np.ndarray,
        hair_maps: np.ndarray | None = None,
    ) -> None:
        job = {
            "steps": steps,
            "streaming": streaming,
            "keypoints": np.asarray(keypoints, dtype=np.float32),
            "sanitize": "none" if self._lab_drive else "constrained",
            "hair_maps": None if hair_maps is None else np.asarray(hair_maps, dtype=np.float32),
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

    def _blend_display_frame(
        self, image: Image.Image, keypoints: np.ndarray | None = None
    ) -> Image.Image:
        """Light temporal EMA so Batch×2 A/B samples don't hard-pop each other.

        Only while the face holds still: blending through a move left the last
        head and hair on screen for 3–4 keys (hair trailing the face).
        """
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        try:
            alpha = _clip_blend(float(self._status.get("frame_blend") or STREAM_TEMPORAL_EMA))
        except (TypeError, ValueError):
            alpha = STREAM_TEMPORAL_EMA
        prev_kps = getattr(self, "_ema_kps", None)
        if keypoints is not None:
            self._ema_kps = np.asarray(keypoints, dtype=np.float32).copy()
            if prev_kps is not None:
                alpha = snap_alpha(alpha, face_pose_delta(prev_kps, self._ema_kps))
        if self._ema_frame is None or self._ema_frame.shape != arr.shape:
            self._ema_frame = arr
            return image
        self._ema_frame = alpha * arr + (1.0 - alpha) * self._ema_frame
        out = np.clip(self._ema_frame, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def _effective_inbetweens(self) -> int:
        return effective_inbetweens(
            self._status.get("interpolate"), self._status.get("inbetweens")
        )

    def _drain_display_queue(self) -> None:
        while True:
            try:
                job = self._display_queue.get_nowait()
            except queue.Empty:
                break
            else:
                try:
                    self._display_queue.task_done()
                except Exception:
                    pass
                if job is None:
                    try:
                        self._display_queue.put_nowait(None)
                    except queue.Full:
                        pass
                    break

    def _enqueue_display(self, item: dict[str, Any]) -> None:
        """Queue one DiT call's keys for playout.

        Whether mids fit is decided at play time, not here: a job arriving while
        the last one is still on screen is the normal case, not a backlog.
        """
        # Keep the newest quarter-second. Older keys would only add lag.
        while True:
            try:
                if int(self._display_queue.qsize()) < PLAYOUT_QUEUE_MAX:
                    break
                old = self._display_queue.get_nowait()
            except queue.Empty:
                break
            if old is None:
                try:
                    self._display_queue.put_nowait(None)
                except queue.Full:
                    pass
                break
            try:
                self._display_queue.task_done()
            except Exception:
                pass
        self._display_queue.put_nowait({**item, "queued_at": time.perf_counter()})

    def _display_worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                job = self._display_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                self._play_display_job(job)
            except Exception as exc:
                print(f"display job failed: {exc}")
            finally:
                try:
                    self._display_queue.task_done()
                except Exception:
                    pass

    def _play_display_job(self, job: dict[str, Any]) -> None:
        """Show one call's keys with mids between each pair, evenly over the
        time that call covers. Batch×2 is two keys here, not two jobs — as two
        jobs the second always looked queued and every mid was skipped."""
        self._display_busy = True
        try:
            keys = job.get("keys") or [(job["image"], job.get("keypoints"))]
            prev = job.get("prev")
            prev_kps = job.get("prev_kps")
            interval = float(job.get("key_interval") or 0.0)
            rate = (1.0 / interval) if interval > 0.0 else 0.0
            wanted = int(job.get("count") or 0)
            count, gap_s = inbetween_pacing(
                rate, wanted, mid_cost_s=float(self._last_interp_s or 0.0)
            )
            with self._lock:
                self._status["inbetweens_live"] = count
            if count < inbetween_pacing(rate, wanted)[0]:
                # Skipped for render cost: nothing re-measures it while mids
                # are off, so let it decay or one slow first mid (cv2 warm-up)
                # turns them off for the whole stream.
                self._last_interp_s = float(self._last_interp_s or 0.0) * 0.8
            for image, keypoints in keys:
                behind = self._display_behind(interval)
                if behind:
                    # A newer call is waiting: drop mids, catch up at 20 fps.
                    count, gap_s = inbetween_pacing(0.0, 0)
                if prev is not None and count > 0:
                    make = inbetween_maker(prev, image)
                    for amount in inbetween_ts(count):
                        if not self._streaming or self._paused or self._display_behind(interval):
                            break
                        started = time.perf_counter()
                        try:
                            mid = make(amount)
                        except Exception as exc:
                            print(f"inbetween failed: {exc}")
                            break
                        cost = time.perf_counter() - started
                        last = float(self._last_interp_s or 0.0)
                        self._last_interp_s = cost if last <= 0.0 else 0.7 * last + 0.3 * cost
                        posed = keypoints
                        if prev_kps is not None and keypoints is not None:
                            posed = lerp_stream_pose(prev_kps, keypoints, amount)
                        self._pace_display(gap_s)
                        self._publish_display_frame(mid, posed, key=False)
                self._pace_display(gap_s)
                self._publish_display_frame(image, keypoints, key=True)
                prev, prev_kps = image, keypoints
        finally:
            self._display_busy = False

    def _display_behind(self, key_interval: float) -> bool:
        """True when playout is really late, not just when the next call is in.

        A big batch plays over several key intervals and the next call often
        lands a few ms before the last key is up — dropping mids for that cost
        ×3/×4 a fifth of them. Late = two calls waiting, or one that has waited
        longer than a key interval.
        """
        q = self._display_queue
        with q.mutex:
            waiting = list(q.queue)
        if not waiting:
            return False
        if len(waiting) >= 2:
            return True
        first = waiting[0]
        if not isinstance(first, dict):
            return True
        queued_at = float(first.get("queued_at") or 0.0)
        if queued_at <= 0.0:
            return True
        return time.perf_counter() - queued_at > max(float(key_interval), 0.05)

    def _pace_display(self, gap_s: float = 0.0) -> None:
        """Hold until the next shown picture is due: ``gap_s`` after the last one
        (at most 20 fps), after a quarter-second hold at the start of a stream."""
        now = time.perf_counter()
        fps_max = (1.0 / gap_s) if gap_s > 0.0 else SHOW_FPS_MAX
        wait, nxt = playout_gap(
            now,
            float(getattr(self, "_playout_next", 0.0) or 0.0),
            fps_max=min(SHOW_FPS_MAX, fps_max),
        )
        self._playout_next = nxt
        if wait <= 0:
            return
        end = time.perf_counter() + wait
        while self._streaming and not self._paused and time.perf_counter() < end:
            time.sleep(min(0.02, end - time.perf_counter()))

    def _shown_fps(self, now: float) -> float:
        """Pictures published over the last second (keys + mids).

        An EMA of 1/gap overshoots on uneven gaps — a Batch×2 pair 2 ms apart
        read as ~170 fps.
        """
        times = getattr(self, "_shown_times", None)
        if times is None:
            times = self._shown_times = deque(maxlen=128)
        times.append(now)
        while times and now - times[0] > SHOWN_FPS_WINDOW_S:
            times.popleft()
        with self._lock:
            if len(times) >= 2 and times[-1] - times[0] > 0.25:
                self._status["show_fps"] = (len(times) - 1) / (times[-1] - times[0])
            return float(self._status.get("show_fps") or 0.0)

    def _note_key_interval(self, now: float, keys: int) -> float:
        """Seconds per generated key, smoothed. Measured on DiT calls, not on
        the display, so pacing and Gen FPS do not feed back into each other."""
        last = float(getattr(self, "_last_call_done_t", 0.0) or 0.0)
        self._last_call_done_t = now
        ema = float(getattr(self, "_key_interval", 0.0) or 0.0)
        dt = now - last
        if last > 0.0 and 0.0 < dt < KEY_INTERVAL_MAX_S:
            per = dt / float(max(1, keys))
            ema = per if ema <= 0.0 else 0.7 * ema + 0.3 * per
            self._key_interval = ema
            with self._lock:
                self._status["gen_fps"] = 1.0 / ema
        return ema

    def _reset_display_clock(self) -> None:
        self._last_call_done_t = 0.0
        self._key_interval = 0.0
        self._shown_times = deque(maxlen=128)
        self._last_interp_s = 0.0
        self._playout_next = 0.0

    def _publish_display_frame(
        self,
        image: Image.Image,
        keypoints: np.ndarray | None,
        *,
        key: bool = True,
    ) -> None:
        if keypoints is not None:
            posed = np.asarray(keypoints, dtype=np.float32)
            self._last_overlay_kps = posed
        else:
            posed = self._last_overlay_kps
        self._last_image = image
        if self._vcam_wanted:
            self._push_virtual_cam(image)
        show_fps = self._shown_fps(time.perf_counter())
        frame = self._frame_payload(image, posed)
        frame["fps"] = show_fps or float(self._status.get("gen_fps") or 0.0)
        self._emit({"type": "frame", **frame})
        if key and self._first_frame_pending:
            self._end_first_frame_wait(message="Streaming")
        if key:
            self._emit({"type": "status", "status": self.status()})

    def _gen_worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                job = self._gen_queue.get(timeout=0.25)
            except queue.Empty:
                self._maybe_offload_after_stop()
                continue
            if job is None:
                break
            streaming = bool(job.get("streaming"))
            if streaming and not self._streaming:
                try:
                    self._gen_queue.task_done()
                except Exception:
                    pass
                self._maybe_offload_after_stop()
                continue
            self._gen_busy = True
            try:
                steps = int(job["steps"])
                keypoints = job["keypoints"]
                images, elapsed = self.engine.generate_batch_from_keypoints(
                    keypoints,
                    num_steps=steps,
                    sanitize=job.get("sanitize", "constrained"),
                    hair_maps=job.get("hair_maps"),
                )
                timings = dict(getattr(self.engine, "last_timings", {}) or {})
                used_batch = getattr(self.engine, "last_target_keypoints_batch", None)
                if used_batch is None:
                    used_batch = np.asarray(keypoints, dtype=np.float32)
                    if used_batch.ndim == 2:
                        used_batch = used_batch[None, ...]

                kps_list: list = []
                for i in range(len(images)):
                    kps_list.append(
                        used_batch[i]
                        if used_batch is not None and i < used_batch.shape[0]
                        else None
                    )
                n = max(1, len(images))
                per = float(elapsed) / float(n) if elapsed > 0 else 0.0
                if streaming:
                    self._note_live_call(len(images), float(elapsed))
                    shown = [
                        self._blend_display_frame(
                            image, kps_list[i] if i < len(kps_list) else None
                        )
                        for i, image in enumerate(images)
                    ]
                    self._on_stream_keys(shown, kps_list, per, timings)
                else:
                    for i, image in enumerate(images):
                        self._on_frame(
                            image,
                            per,
                            timings,
                            kps_list[i] if i < len(kps_list) else None,
                            streaming=False,
                            schedule_next=False,
                        )
            except Exception as exc:
                if streaming:
                    self._streaming = False
                    self._paused = False
                    self._frame_in_flight = False
                    self._offload_pending = True
                    self._end_first_frame_wait()
                    self._set_status(
                        streaming=False,
                        paused=False,
                        busy=False,
                        error=str(exc),
                        message="Stream error",
                    )
                else:
                    self._frame_in_flight = False
                    self._set_status(busy=False, error=str(exc), message="Generate failed")
            finally:
                self._gen_busy = False
                if not streaming:
                    self._frame_in_flight = False
                try:
                    self._gen_queue.task_done()
                except Exception:
                    pass
                self._maybe_offload_after_stop()

    def _note_timing(self, elapsed: float, timings: dict, *, streaming: bool) -> str:
        dit_fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        with self._lock:
            den = float(timings.get("denoise_s", timings.get("denoise", 0)) or 0)
            dec = float(timings.get("decode_s", timings.get("decode", 0)) or 0)
            timing = f"denoise {den:.2f}s | decode {dec:.2f}s" if timings else ""
            batch_n = int(float(timings.get("batch", 0) or 0))
            if batch_n > 1:
                timing = f"{timing} | batch×{batch_n}" if timing else f"batch×{batch_n}"
            if dit_fps > 0:
                timing = f"{timing} | gpu {dit_fps:.1f}" if timing else f"gpu {dit_fps:.1f}"
            message = "Streaming"
            if streaming and self._paused:
                message = "Stream paused"
            elif not streaming:
                message = "Generated"
            update = {
                "timing": timing,
                "busy": False if not streaming else self._status.get("busy", False),
                "message": message,
            }
            if not streaming:
                update["gen_fps"] = dit_fps
                update["show_fps"] = dit_fps
            self._status.update(update)
        return timing

    def _on_stream_keys(
        self,
        images: list[Image.Image],
        keypoints: list[np.ndarray | None],
        elapsed: float,
        timings: dict,
        *,
        schedule_next: bool = True,
    ) -> None:
        """One streaming DiT call is done: start the next, queue these keys."""
        self._note_timing(elapsed, timings, streaming=True)
        key_interval = self._note_key_interval(time.perf_counter(), len(images))
        keys = [
            (
                image,
                None
                if i >= len(keypoints) or keypoints[i] is None
                else np.asarray(keypoints[i], dtype=np.float32).copy(),
            )
            for i, image in enumerate(images)
        ]
        prev = self._inbetween_prev
        prev_kps = self._inbetween_prev_kps
        self._inbetween_prev = keys[-1][0].copy()
        self._inbetween_prev_kps = keys[-1][1]
        # Next DiT call starts now. Optical flow cannot steal generate time.
        if schedule_next:
            self._frame_in_flight = False
            if self._streaming and not self._paused:
                self._schedule_next_frame()
        self._enqueue_display(
            {
                "keys": keys,
                "prev": prev,
                "prev_kps": prev_kps,
                "count": self._effective_inbetweens(),
                "key_interval": key_interval,
            }
        )
        self._emit({"type": "status", "status": self.status()})

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
        if streaming:
            self._on_stream_keys(
                [image],
                [keypoints],
                elapsed,
                timings,
                schedule_next=schedule_next is not False,
            )
            return
        timing = self._note_timing(elapsed, timings, streaming=False)
        dit_fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        self._last_image = image
        if keypoints is not None:
            self._last_overlay_kps = np.asarray(keypoints, dtype=np.float32)
        if self._vcam_wanted:
            self._push_virtual_cam(image)
        frame = self._frame_payload(image, self._last_overlay_kps)
        frame["elapsed"] = elapsed
        frame["fps"] = dit_fps
        frame["timing"] = timing
        self._emit({"type": "frame", **frame})
        self._emit({"type": "status", "status": self.status()})

    def _push_virtual_cam(self, image: Image.Image) -> None:
        from .vcam_device import DEVICE_NAME
        from .virtual_cam import VCAM_FPS, get_virtual_cam

        vcam = get_virtual_cam()
        if not vcam.active:
            try:
                w, h = self._vcam_frame_size()
                device = vcam.start(w, h, fps=VCAM_FPS, source=self._vcam_source)
                self._set_status(
                    virtual_cam=True,
                    virtual_cam_device=device or DEVICE_NAME,
                    virtual_cam_error="",
                    virtual_cam_width=w,
                    virtual_cam_height=h,
                )
            except Exception as exc:
                self._vcam_wanted = False
                self._set_status(
                    virtual_cam=False,
                    virtual_cam_device="",
                    virtual_cam_error=str(exc),
                    virtual_cam_width=0,
                    virtual_cam_height=0,
                    error=str(exc),
                )
                return
        vcam.send(image)
        err = vcam.error
        if err and self.status().get("virtual_cam"):
            self._vcam_wanted = False
            self._set_status(
                virtual_cam=False,
                virtual_cam_device="",
                virtual_cam_error=err,
                virtual_cam_width=0,
                virtual_cam_height=0,
                error=err,
            )

    def _lab_hair_wh(self, frame: dict[str, Any] | None = None) -> tuple[int, int]:
        from .lab_harness import frame_image_wh

        lab = frame_image_wh(frame)
        if lab is not None:
            self._lab_image_wh = lab
            return lab
        image = self._last_image
        if image is not None:
            return image.size
        stored = getattr(self, "_lab_image_wh", None)
        if stored:
            return int(stored[0]), int(stored[1])
        return (0, 0)

    def _adopt_lab_hair(self, frame: dict[str, Any] | None = None) -> bool:
        """Copy Track Lab hair polygons. Do not re-segment on the desk."""
        from .lab_harness import hair_from_frame, lab as lab_harness

        packet = frame if isinstance(frame, dict) else lab_harness.frame()
        if not self._lab_overlay_current(packet):
            return False
        self._note_lab_generation(packet)
        w, h = self._lab_hair_wh(packet)
        hair = hair_from_frame(packet, width=w, height=h)
        if hair is None and frame is None:
            status = lab_harness.status(merge_frame=True)
            if isinstance(status, dict) and self._lab_overlay_current(status):
                w, h = self._lab_hair_wh(status)
                try:
                    w = w or int(status.get("width") or 0)
                    h = h or int(status.get("height") or 0)
                except (TypeError, ValueError):
                    pass
                hair = hair_from_frame(status, width=w, height=h)
                self._note_lab_generation(status)
        if not hair:
            return False
        self._last_lab_hair = hair
        if not self._lab_drive or getattr(self, "_lab_rest_hair", None) is None:
            self._lab_rest_hair = [dict(seg) for seg in hair]
        self._hair_capture_done = True
        self._hair_rig = None
        return True

    def _lab_harness_online(self) -> bool:
        try:
            from .lab_harness import lab as lab_harness

            return bool(lab_harness.status(merge_frame=False).get("online"))
        except Exception:
            return False

    def _maybe_capture_hair(self, rest_keypoints: np.ndarray | None = None) -> None:
        """Prefer Track Lab hair. Desk animeseg is only the offline fallback."""
        if self._hair_rig is not None or self._hair_capture_done:
            return
        if self._last_lab_hair:
            self._hair_capture_done = True
            return
        if getattr(self, "_lab_drive", False) and getattr(self, "_lab_overlay_gen", None) is not None:
            self._hair_capture_done = True
            return
        self._capture_character_hair_mesh(rest_keypoints=rest_keypoints)

    def _ensure_hair_overlay_tracker(self):
        if self._hair_overlay_tracker is not None:
            return self._hair_overlay_tracker
        path = resolve_hair_weights(None)
        if path is None:
            print("Hair overlay: animeseg_hair3.pt not found")
            return None
        try:
            # One-shot capture only. Live frames follow the mesh with offsets.
            self._hair_overlay_tracker = create_hair_tracker(path, device="cpu")
            print(
                f"Hair overlay model: {path.name} ({self._hair_overlay_tracker.method})"
            )
        except Exception as exc:
            print(f"Hair overlay load failed: {exc}")
            self._hair_overlay_tracker = None
        return self._hair_overlay_tracker

    def _release_hair_tracker(self) -> None:
        self._hair_overlay_tracker = None

    def character_fit(self) -> dict[str, Any]:
        """Hair, skeleton, and limiter boxes in still pixels for the fit canvas."""
        from .character_fit import build_fit_view

        image = self._last_image
        if image is None:
            raise RuntimeError("Create a character before fitting it")
        hair = self._rest_hair_segments() or self._last_lab_hair or []
        return build_fit_view(
            width=int(image.size[0]),
            height=int(image.size[1]),
            keypoints=self._last_overlay_kps,
            hair_norm=hair,
            box=self._status.get("travel_box"),
        )

    def paint_character_hair(
        self,
        points: list,
        *,
        radius: float,
        part: str,
        erase: bool = False,
    ) -> dict[str, Any]:
        """Stamp a brush stroke onto the character hair and keep it."""
        from .character_fit import (
            norm_hair_to_pixels,
            pixels_hair_to_norm,
            update_character_fit,
        )
        from .hair_edit import HAIR_PARTS, stamp_hair_stroke

        image = self._last_image
        if image is None:
            raise RuntimeError("Create a character before painting hair")
        if not points:
            raise ValueError("Paint a stroke on the hair")
        chosen = str(part or "hair_middle")
        if chosen not in HAIR_PARTS:
            raise ValueError("Pick middle, left, or right hair")
        width, height = image.size
        current = norm_hair_to_pixels(self._rest_hair_segments() or self._last_lab_hair or [], width, height)
        stamped = stamp_hair_stroke(
            current,
            points,
            radius=max(1.0, min(80.0, float(radius))),
            part=chosen,
            width=width,
            height=height,
            erase=bool(erase),
        )
        norm = pixels_hair_to_norm(stamped, width, height)
        self._last_lab_hair = norm
        self._lab_rest_hair = [dict(seg) for seg in norm]
        self._hair_capture_done = True
        self._hair_rig = None
        ident = str(self._status.get("character_id") or "")
        if ident:
            update_character_fit(ident, {"hair": norm})
        lab_hair = []
        for seg in stamped:
            poly = []
            for vertex in seg.get("polygon") or []:
                if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                    px, py = self._desk_px_to_lab(float(vertex[0]), float(vertex[1]))
                    poly.append([round(px, 1), round(py, 1)])
            if len(poly) >= 3:
                lab_hair.append({"class": seg.get("class"), "polygon": poly})
        try:
            ack = self._lab_ack("set_hair", {"hair": lab_hair})
            packet = self._lab_packet_from_ack(ack)
            self._adopt_lab_hair(packet)
        except Exception as exc:
            print(f"Hair paint kept on the desk; Track Lab did not store it: {exc}")
        if self._last_image is not None:
            self._emit(
                {
                    "type": "frame",
                    **self._frame_payload(self._last_image, self._last_overlay_kps),
                }
            )
        return self.character_fit()

    def move_character_skeleton(self, idx: int, x: float, y: float) -> dict[str, Any]:
        """Place one rest skeleton joint. Face points stay where the fit left them."""
        from .character_fit import SKELETON_LABELS, update_character_fit

        image = self._last_image
        kps = self._last_overlay_kps
        if image is None or kps is None:
            raise RuntimeError("Create a character before moving the skeleton")
        slot = int(idx)
        if slot not in SKELETON_LABELS:
            raise ValueError("Only the neck, shoulders, elbows, and chest can be moved here")
        width, height = image.size
        nx, ny = pixels_to_normalized(float(x), float(y), width, height)
        edited = np.asarray(kps, dtype=np.float32).copy()
        edited[slot, 0] = nx
        edited[slot, 1] = ny
        edited[slot, 2] = max(float(edited[slot, 2]), 0.85)
        edited[slot, 3] = 1.0
        self._install_fit_keypoints(edited)
        try:
            px, py = self._desk_px_to_lab(float(x), float(y))
            ack = self._lab_ack("set_skeleton_point", {"id": slot, "x": float(px), "y": float(py)})
            packet = self._lab_packet_from_ack(ack)
            self.adopt_lab_overlay(packet, emit=False)
            merged = (
                np.asarray(self._last_overlay_kps, dtype=np.float32).copy()
                if self._last_overlay_kps is not None
                else edited
            )
            merged[slot, 0] = nx
            merged[slot, 1] = ny
            merged[slot, 2] = max(float(merged[slot, 2]), 0.85)
            merged[slot, 3] = 1.0
            self._install_fit_keypoints(merged)
        except Exception as exc:
            print(f"Skeleton move kept on the desk; Track Lab did not store it: {exc}")
        ident = str(self._status.get("character_id") or "")
        if ident and self._last_overlay_kps is not None:
            body = []
            for joint in SKELETON_LABELS:
                body.append(
                    {
                        "id": joint,
                        "x": round(float(self._last_overlay_kps[joint, 0]), 5),
                        "y": round(float(self._last_overlay_kps[joint, 1]), 5),
                    }
                )
            update_character_fit(ident, {"skeleton": body})
        if self._last_image is not None:
            self._emit(
                {
                    "type": "frame",
                    **self._frame_payload(self._last_image, self._last_overlay_kps),
                }
            )
        return self.character_fit()

    def move_character_point(self, idx: int, x: float, y: float) -> dict[str, Any]:
        """Place one rest tracking point: a face point, an iris, or a skeleton joint.

        The rest mesh is saved into the ``.vtm`` and Track Lab moves the same
        rest point, dropping any overlay nudge on it so the edit is not
        applied twice.
        """
        from .character_fit import POINT_SLOTS, SKELETON_LABELS

        slot = int(idx)
        if slot in SKELETON_LABELS:
            return self.move_character_skeleton(slot, x, y)
        image = self._last_image
        kps = self._last_overlay_kps
        if image is None or kps is None:
            raise RuntimeError("Create a character before moving its points")
        if slot not in POINT_SLOTS:
            raise ValueError(f"Point {slot} cannot be moved here")
        width, height = image.size
        nx, ny = pixels_to_normalized(float(x), float(y), width, height)
        edited = np.asarray(kps, dtype=np.float32).copy()
        edited[slot, 0] = nx
        edited[slot, 1] = ny
        edited[slot, 2] = max(float(edited[slot, 2]), 0.85)
        edited[slot, 3] = 1.0
        self._install_fit_keypoints(edited)
        try:
            px, py = self._desk_px_to_lab(float(x), float(y))
            self._lab_ack("set_rest_point", {"id": slot, "x": float(px), "y": float(py)})
        except Exception as exc:
            print(f"Point move kept on the desk; Track Lab did not store it: {exc}")
        if self._last_image is not None:
            self._emit(
                {
                    "type": "frame",
                    **self._frame_payload(self._last_image, self._last_overlay_kps),
                }
            )
        return self.character_fit()

    def _install_fit_keypoints(self, keypoints: np.ndarray) -> None:
        from .character_fit import replace_pack_keypoints

        kps = np.asarray(keypoints, dtype=np.float32).copy()
        self._last_overlay_kps = kps
        self._driven_keypoints = kps.copy()
        self._last_good_keypoints = kps.copy()
        try:
            self.engine.adopt_ref_keypoints(kps, persist=False, pose_source="fit")
        except Exception as exc:
            print(f"Fit pose did not install: {exc}")
        path = self._ref_path
        if path is not None and path.suffix.lower() == ".vtm" and path.is_file():
            try:
                replace_pack_keypoints(path, kps)
            except Exception as exc:
                print(f"Fit pose did not save into the character: {exc}")

    def _apply_character_fit(self) -> bool:
        """Put a saved hair paint and skeleton back after Track Lab re-detects."""
        from .character_fit import norm_hair_to_pixels, read_character_fit

        ident = str(self._status.get("character_id") or "")
        if not ident:
            return False
        saved = read_character_fit(ident)
        if not saved:
            return False
        changed = False
        skeleton = saved.get("skeleton")
        if isinstance(skeleton, list) and self._last_overlay_kps is not None:
            kps = np.asarray(self._last_overlay_kps, dtype=np.float32).copy()
            moved = False
            for row in skeleton:
                if not isinstance(row, dict):
                    continue
                try:
                    slot = int(row.get("id", -1))
                    nx = float(row.get("x"))
                    ny = float(row.get("y"))
                except (TypeError, ValueError):
                    continue
                if not (31 <= slot <= 36):
                    continue
                if (
                    abs(float(kps[slot, 0]) - nx) > 1e-5
                    or abs(float(kps[slot, 1]) - ny) > 1e-5
                    or float(kps[slot, 3]) < 0.5
                ):
                    moved = True
                kps[slot, 0] = nx
                kps[slot, 1] = ny
                kps[slot, 2] = max(float(kps[slot, 2]), 0.85)
                kps[slot, 3] = 1.0
                try:
                    px, py = self._norm_to_lab_px(nx, ny)
                    self._lab_ack("set_skeleton_point", {"id": slot, "x": px, "y": py})
                except Exception:
                    pass
            if moved:
                self._install_fit_keypoints(kps)
                changed = True
        hair = saved.get("hair")
        if isinstance(hair, list) and hair:
            self._last_lab_hair = hair
            self._lab_rest_hair = [dict(seg) for seg in hair]
            self._hair_capture_done = True
            self._hair_rig = None
            changed = True
            image = self._last_image
            lab = getattr(self, "_lab_image_wh", None)
            if image is not None:
                lw, lh = (int(lab[0]), int(lab[1])) if lab else image.size
                try:
                    self._lab_ack(
                        "set_hair",
                        {"hair": norm_hair_to_pixels(hair, lw, lh)},
                    )
                except Exception:
                    pass
        return changed

    def _norm_to_lab_px(self, x: float, y: float) -> tuple[float, float]:
        image = self._last_image
        lab = getattr(self, "_lab_image_wh", None)
        if lab:
            width, height = int(lab[0]), int(lab[1])
        elif image is not None:
            width, height = image.size
        else:
            return float(x), float(y)
        return (float(x) + 1.0) * 0.5 * float(width), (float(y) + 1.0) * 0.5 * float(height)

    def _capture_character_hair_mesh(
        self, rest_keypoints: np.ndarray | None = None
    ) -> None:
        """Run animeseg once on the reference image and bind a follow rig."""
        if self._hair_rig is not None or self._hair_capture_done:
            return
        image = self._last_image
        rest = rest_keypoints if rest_keypoints is not None else self._last_overlay_kps
        if image is None or rest is None:
            return
        self._hair_capture_done = True
        tracker = self._ensure_hair_overlay_tracker()
        if tracker is None:
            return
        try:
            rgb = np.asarray(image.convert("RGB"))
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            with self._hair_overlay_lock:
                segs_px = tracker.detect(bgr)
        except Exception as exc:
            print(f"Hair mesh capture failed: {exc}")
            segs_px = None
        finally:
            self._release_hair_tracker()
        if not segs_px:
            print("Hair mesh capture: no segments — overlay/maps stay empty")
            return
        w, h = image.size
        segs_norm = self._pixel_hair_to_norm(segs_px, w, h)
        self._hair_rig = build_hair_rig(segs_norm, np.asarray(rest, dtype=np.float32))
        n = len(self._hair_rig.parts) if self._hair_rig is not None else 0
        print(f"Hair mesh captured ({n} parts) — live frames follow pose offsets")

    def _frame_payload(
        self, image: Image.Image | None, keypoints: np.ndarray | None
    ) -> dict[str, Any]:
        st = self.status()
        show_mesh = bool(st.get("show_mesh"))
        show_hair = bool(st.get("show_hair", True))
        show_limiters = bool(st.get("show_limiters"))
        overlay_on = show_mesh or show_hair
        preview = getattr(self, "_travel_preview_kps", None)
        mesh_kps = preview if preview is not None else keypoints
        display = image
        if display is not None and overlay_on:
            try:
                display = display.copy()
                if show_mesh and mesh_kps is not None:
                    display = draw_keypoint_mesh(
                        display,
                        mesh_kps,
                        skeleton_lost=bool(self._body_lost),
                        mouth_snapped=bool(self._mouth_snapped),
                        show_outline=bool(st.get("show_outline", True)),
                        show_brows=bool(st.get("show_brows", True)),
                        show_eyes=bool(st.get("show_eyes", True)),
                        show_nose=bool(st.get("show_nose", True)),
                        show_mouth=bool(st.get("show_mouth", True)),
                        show_iris=bool(st.get("show_iris_overlay", True)),
                        show_skeleton=bool(st.get("show_skeleton", True)),
                    )
            except Exception:
                display = image
            if show_hair:
                try:
                    hair_segs = self._followed_hair_segments(mesh_kps)
                    if hair_segs:
                        display = draw_hair_overlay(display, hair_segs)
                except Exception as exc:
                    print(f"Hair overlay draw failed: {exc}")
        if display is not None and show_limiters:
            try:
                raw_box = st.get("travel_box")
                box = dict(raw_box) if isinstance(raw_box, dict) else {}
                box["enabled"] = True
                display = draw_travel_box(display, self._limiter_rest(), box)
            except Exception as exc:
                print(f"Limiter overlay draw failed: {exc}")
        payload: dict[str, Any] = {
            "image": _image_to_jpeg_b64(display) if display is not None else None,
            "width": int(display.width) if display is not None else 0,
            "height": int(display.height) if display is not None else 0,
            "keypoints": None,
        }
        if keypoints is not None:
            payload["keypoints"] = np.asarray(keypoints, dtype=np.float32).tolist()
        return payload

    def current_frame_event(self) -> dict[str, Any] | None:
        """Still on the desk right now, for a client that connected after boot."""
        image = getattr(self, "_last_image", None)
        if image is None:
            return None
        return {
            "type": "frame",
            **self._frame_payload(image, getattr(self, "_last_overlay_kps", None)),
        }

    def mesh_press(self, x: float, y: float) -> None:
        st = self.status()
        if not st.get("show_mesh") or self._last_image is None:
            return
        self._travel_preview_kps = None
        allowed = overlay_visible_slots(st)
        if not allowed:
            return
        with self._lock:
            kps = self._last_overlay_kps
            image = self._last_image
            if kps is None or image is None:
                return
            w, h = image.size
            idx = nearest_keypoint(kps, x, y, w, h, include=allowed)
            if idx is None or not (0 <= int(idx) < NUM_KEYPOINTS):
                self._drag_kp_idx = None
                self._drag_slots = set()
                self._drag_base_kps = None
                self._drag_xy = None
                return
            idx = int(idx)
            slots = {idx}
            base = np.asarray(kps, dtype=np.float32).copy()
            self._drag_kp_idx = idx
            self._drag_slots = slots
            self._drag_base_kps = base
            self._drag_xy = (float(x), float(y))
            # Lab nudges ride on live tracking. Only freeze the whole overlay
            # when the pose is already held or the old webcam path is driving.
            if not self._lab_drive or self._pose_frozen:
                self._mesh_edited = True

    def mesh_drag(self, x: float, y: float) -> None:
        emit_image: Image.Image | None = None
        emit_kps: np.ndarray | None = None
        try:
            with self._lock:
                slots = set(self._drag_slots)
                lab = bool(self._lab_drive) and not bool(self._pose_frozen)
                if not slots or self._last_image is None:
                    return
                kps = (
                    np.asarray(self._last_overlay_kps, dtype=np.float32).copy()
                    if self._last_overlay_kps is not None
                    else None
                )
                if kps is None or kps.shape[0] < NUM_KEYPOINTS:
                    return
                w, h = self._last_image.size
                nx, ny = pixels_to_normalized(x, y, w, h)
                for j in slots:
                    if 0 <= j < NUM_KEYPOINTS:
                        kps[j, 0] = nx
                        kps[j, 1] = ny
                        kps[j, 2] = max(float(kps[j, 2]), 0.85)
                        kps[j, 3] = 1.0
                self._last_overlay_kps = kps
                self._driven_keypoints = kps.copy()
                if not lab:
                    self._mesh_edited = True
                self._drag_xy = (float(x), float(y))
                now = time.monotonic()
                if now - self._last_mesh_emit_t >= 0.04:
                    self._last_mesh_emit_t = now
                    emit_image = self._last_image
                    emit_kps = kps
            if emit_image is not None and emit_kps is not None:
                self._emit({"type": "frame", **self._frame_payload(emit_image, emit_kps)})
        except Exception as exc:
            print(f"mesh drag failed: {exc}")

    def mesh_release(self) -> None:
        """End a point drag. Lab nudges ride on tracking; frozen poses keep the edit."""
        try:
            with self._lock:
                frozen = bool(self._pose_frozen)
                lab = bool(self._lab_drive)
                slots = set(self._drag_slots)
                drag_xy = self._drag_xy
                base = (
                    None
                    if self._drag_base_kps is None
                    else np.asarray(self._drag_base_kps, dtype=np.float32).copy()
                )
                edited = (
                    None
                    if self._last_overlay_kps is None
                    else np.asarray(self._last_overlay_kps, dtype=np.float32).copy()
                )
                image = self._last_image
                # Keep the grabbed slot pinned until set_point lands, otherwise
                # the live poll snaps it back to the unoffset track.
                if not lab:
                    self._drag_kp_idx = None
                    self._drag_slots = set()
                    self._drag_base_kps = None
                    self._drag_xy = None
            moved = False
            if slots and base is not None and edited is not None:
                for i in slots:
                    if abs(float(edited[i, 0] - base[i, 0])) > 1e-5:
                        moved = True
                        break
                    if abs(float(edited[i, 1] - base[i, 1])) > 1e-5:
                        moved = True
                        break
            if lab and moved and slots:
                idx = int(next(iter(slots)))
                px = py = None
                if drag_xy is not None:
                    px, py = float(drag_xy[0]), float(drag_xy[1])
                elif image is not None and edited is not None:
                    w, h = image.size
                    px = (float(edited[idx, 0]) + 1.0) * 0.5 * float(w)
                    py = (float(edited[idx, 1]) + 1.0) * 0.5 * float(h)
                if px is not None and py is not None:
                    self._nudge_lab_point(idx, px, py)
            if lab:
                with self._lock:
                    self._drag_kp_idx = None
                    self._drag_slots = set()
                    self._drag_base_kps = None
                    self._drag_xy = None
            if frozen:
                with self._lock:
                    self._mesh_edited = True
                    if edited is not None:
                        self._driven_keypoints = edited.copy()
                if image is not None and edited is not None:
                    self._emit({"type": "frame", **self._frame_payload(image, edited)})
                return
            if lab:
                with self._lock:
                    self._mesh_edited = False
                driven = None
                if self._tracking:
                    try:
                        driven = self._lab_overlay_keypoints()
                    except Exception:
                        driven = None
                overlay = driven if driven is not None else edited
                if image is not None and overlay is not None:
                    self._emit({"type": "frame", **self._frame_payload(image, overlay)})
                return
            rest = getattr(self.engine, "_ref_keypoints", None)
            if not slots or base is None or edited is None or rest is None or not moved:
                with self._lock:
                    self._mesh_edited = False
                return
            new_rest = apply_overlay_drag_to_rest(rest, base, edited, slots)
            saved = self.engine.adopt_ref_keypoints(new_rest, persist=True)
            driven = None
            if self._tracking:
                try:
                    driven = self._retarget_live_to_character()
                except Exception:
                    driven = None
            with self._lock:
                self._mesh_edited = False
                if driven is None:
                    self._last_overlay_kps = np.asarray(new_rest, dtype=np.float32).copy()
                    self._driven_keypoints = self._last_overlay_kps.copy()
                overlay = self._last_overlay_kps
            if image is not None and overlay is not None:
                self._emit({"type": "frame", **self._frame_payload(image, overlay)})
            if saved is not None:
                print(
                    f"Mesh edits kept — {saved.name} is the new rest pose. "
                    "Tracking follows this layout. Double-click the preview to undo."
                )
        except Exception as exc:
            print(f"mesh release failed: {exc}")
            with self._lock:
                self._mesh_edited = False
                self._drag_kp_idx = None
                self._drag_slots = set()
                self._drag_base_kps = None
                self._drag_xy = None

    def mesh_reset(self) -> None:
        try:
            with self._lock:
                frozen = bool(self._pose_frozen)
                lab = bool(self._lab_drive)
                auto = (
                    None
                    if self._freeze_auto_kps is None
                    else np.asarray(self._freeze_auto_kps, dtype=np.float32).copy()
                )
                self._drag_kp_idx = None
                self._drag_slots = set()
                self._drag_base_kps = None
                self._drag_xy = None
            if lab:
                try:
                    self._lab_ack("reset_points", {})
                except Exception as exc:
                    print(f"lab reset_points failed: {exc}")
            if frozen and auto is not None:
                with self._lock:
                    self._mesh_edited = True
                    self._last_overlay_kps = auto
                    self._driven_keypoints = auto.copy()
                    image = self._last_image
                    overlay = auto
                if image is not None and overlay is not None:
                    self._emit({"type": "frame", **self._frame_payload(image, overlay)})
                return
            if lab:
                with self._lock:
                    self._mesh_edited = False
                driven = None
                if self._tracking:
                    try:
                        driven = self._lab_overlay_keypoints()
                    except Exception:
                        driven = None
                if driven is not None and self._last_image is not None:
                    self._emit({"type": "frame", **self._frame_payload(self._last_image, driven)})
                print("Overlay nudges reset to the original Track Lab points")
                return
            with self._lock:
                self._mesh_edited = True
            rest = self.engine.restore_session_ref_keypoints()
            with self._lock:
                self._mesh_edited = False
                self._last_overlay_kps = np.asarray(rest, dtype=np.float32).copy()
                self._driven_keypoints = self._last_overlay_kps.copy()
                image = self._last_image
                overlay = self._last_overlay_kps
            if image is not None and overlay is not None:
                self._emit(
                    {
                        "type": "frame",
                        **self._frame_payload(image, overlay),
                    }
                )
            print("Mesh edits reset to the reference pose from this session")
        except Exception as exc:
            print(f"mesh reset failed: {exc}")
            with self._lock:
                self._mesh_edited = False
                self._drag_kp_idx = None
                self._drag_slots = set()
                self._drag_base_kps = None
                self._drag_xy = None

    def freeze_pose(self) -> dict[str, Any]:
        """Hold the current overlay so points can be dragged without tracking overwriting them."""
        overlay = self._last_overlay_kps
        if overlay is None:
            overlay = self._current_keypoints()
        if overlay is None:
            return self.status()
        held = np.asarray(overlay, dtype=np.float32).copy()
        ctrl = self._prev_controls
        show = bool(self.status().get("show_mesh"))
        with self._lock:
            self._pose_frozen = True
            self._mesh_edited = True
            self._freeze_auto_kps = held.copy()
            self._freeze_params = params_from_controls(ctrl)
            self._freeze_roll = float(getattr(ctrl, "head_roll_deg", 0.0) or 0.0) if ctrl is not None else 0.0
            self._last_overlay_kps = held
            self._driven_keypoints = held.copy()
            image = self._last_image
        if image is not None:
            self._emit({"type": "frame", **self._frame_payload(image, held)})
        patch = {
            "pose_frozen": True,
            "message": "Pose frozen — drag overlay points",
            "track_message": "Frozen",
        }
        if not show:
            patch["show_mesh"] = True
            for key in OVERLAY_PART_KEYS:
                patch[key] = True
            patch["show_hair"] = True
        self._set_status(**patch)
        return self.status()

    def unfreeze_pose(self) -> dict[str, Any]:
        with self._lock:
            self._pose_frozen = False
            self._mesh_edited = False
            self._freeze_auto_kps = None
            self._freeze_params = None
            self._freeze_roll = 0.0
            self._drag_kp_idx = None
            self._drag_slots = set()
            self._drag_base_kps = None
            self._drag_xy = None
        driven = None
        if self._tracking:
            try:
                driven = (
                    self._lab_overlay_keypoints()
                    if self._lab_drive
                    else self._retarget_live_to_character()
                )
            except Exception:
                driven = None
        if driven is not None and self._last_image is not None:
            self._emit({"type": "frame", **self._frame_payload(self._last_image, driven)})
        self._set_status(
            pose_frozen=False,
            pose_key_count=len(self._pose_keys),
            message="Tracking resumed",
            track_message="Tracking on" if self._tracking else "Tracking off",
        )
        return self.status()

    def _track_poll_loop(self) -> None:
        while not self._track_stop.is_set():
            time.sleep(0.05)
            if not self._tracking:
                continue
            try:
                if self._lab_drive:
                    # #region agent log
                    _now = time.time()
                    if debug_log.ENABLED and _now - float(getattr(self, "_dbg_t", 0.0)) >= 0.5:
                        self._dbg_t = _now
                        _kps = self._last_overlay_kps
                        _prev = getattr(self, "_dbg_kps", None)
                        _motion = 0.0
                        if _kps is not None and _prev is not None and getattr(_prev, "shape", None) == _kps.shape:
                            _motion = float(np.max(np.abs(_kps[:, :2] - _prev[:, :2])))
                        if _kps is not None:
                            self._dbg_kps = np.asarray(_kps, dtype=np.float32).copy()
                        debug_log.log(
                            "E",
                            "stream.py:_track_poll_loop",
                            "desk overlay",
                            {
                                "streaming": bool(self._streaming),
                                "frozen": bool(self._pose_frozen),
                                "preview_age": round(_now - float(self._last_live_preview_t or 0.0), 3),
                                "motion": round(_motion, 4),
                                "has_kps": _kps is not None,
                            },
                        )
                    # #endregion
                    with self._lock:
                        mesh_frozen = bool(self._pose_frozen)
                    if not self._streaming and not mesh_frozen:
                        self._last_live_preview_t = time.time()
                        driven = self._lab_overlay_keypoints()
                        if driven is not None and self._last_image is not None:
                            self._emit(
                                {
                                    "type": "frame",
                                    **self._frame_payload(self._last_image, driven),
                                }
                            )
                    self._set_status(
                        track_message="Frozen" if self._pose_frozen else "Tracking on",
                        body_label="",
                    )
                    continue
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
                    with self._lock:
                        mesh_frozen = bool(self._mesh_edited or self._pose_frozen)
                    if not self._streaming and not mesh_frozen:
                        self._last_live_preview_t = time.time()
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
                if self._pose_frozen:
                    msg = "Frozen"
                else:
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
            self.stop_virtual_cam()
        except Exception:
            pass
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
        try:
            self._display_queue.put_nowait(None)
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
