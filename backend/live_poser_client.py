"""Headless LivePoser tracker for VTM Noble.

Embeds tools/live-poser camera + OSF → KEYPOINT_SCHEMA (37,4) bridge.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import ensure_import_paths, live_poser_dir, torch_train_dir

ensure_import_paths()
LIVE_POSER_DIR = live_poser_dir()
TORCH_TRAIN_DIR = torch_train_dir()

from bridge import BridgeFrame  # noqa: E402
from calibration import CenterCalibration, RelativePose  # noqa: E402
from cameras import (  # noqa: E402
    CameraCapture,
    CameraInfo,
    ensure_com,
    ensure_osf_on_path,
    list_cameras as lp_list_cameras,
    resolve_openseeface,
    uninit_com,
)
from label_schema import normalize_mouth_osf_map  # noqa: E402
from tracking_filters import FilterSettings, MotionSmoother  # noqa: E402

from .tracking import (  # noqa: E402
    BodyTrack,
    BodyTracker,
    FaceTrack,
    FaceTracker,
    HairTracker,
    IrisTrack,
    IrisTracker,
    merge_tracks,
)

# Reuse helpers from live_poser without importing the Tk UI class.
from live_poser import (  # noqa: E402
    MAX_TRACK_WIDTH,
    build_tracker,
    detection_reason,
    fit_frame,
    frame_health,
    pick_default_camera,
)

try:
    from live_poser import close_tracker  # noqa: E402
except ImportError:
    def close_tracker(tracker) -> None:  # type: ignore[misc]
        """Fallback if vendor live_poser was re-synced without close_tracker."""
        if tracker is None:
            return
        try:
            if hasattr(tracker, "close"):
                tracker.close()
        except Exception:
            pass
        for attr in ("session", "sessions", "gaze_model", "detection", "retinaface", "retinaface_scan"):
            try:
                setattr(tracker, attr, None if attr != "sessions" else [])
            except Exception:
                pass

from utils.coordinate_frames import (  # noqa: E402
    COORD_NORM_CROP,
    DEFAULT_IMAGE_SIZE,
    CropRect,
    webcam_pixels_to_norm_crop,
)

NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4
FACE_MAX_AGE_S = 3.0
LIVE_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
CAPTURE_READ_FAIL_LIMIT = 3
CAPTURE_BLACK_FRAME_LIMIT = 8
CAPTURE_REOPEN_BACKOFF_S = (0.25, 0.75, 1.5, 3.0)
# DirectShow virtual cams (DroidCam) often keep the device locked briefly after release.
CAPTURE_RELEASE_SETTLE_S = 0.85
CAPTURE_STOP_JOIN_S = 8.0
CAPTURE_OPEN_ATTEMPTS = 3
# Face-detector recovery when camera frames look fine but OSF finds nobody.
# Keep cooldown high — each rebuild used to leak ONNX sessions (~hundreds of MB).
NO_FACE_REBUILD_S = 4.0
NO_FACE_REBUILD_COOLDOWN_S = 30.0
NO_FACE_REOPEN_S = 12.0
NO_FACE_REBUILD_MAX = 3


def list_cameras() -> list[tuple[int, str]]:
    """Return (index, name) pairs for the UI combobox."""
    ensure_com()
    cams = lp_list_cameras()
    return [(c.index, c.name) for c in cams] or [(0, "Camera 0")]


def list_camera_infos() -> list[CameraInfo]:
    ensure_com()
    return lp_list_cameras()


def pick_default_camera_index(cameras: list[tuple[int, str]]) -> int:
    infos = [CameraInfo(index=i, name=n, backend="unknown") for i, n in cameras]
    idx = pick_default_camera(infos)
    if 0 <= idx < len(cameras):
        return cameras[idx][0]
    return cameras[0][0] if cameras else 0


def keypoints_to_norm_crop(
    kps_px: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    image_size: int = LIVE_IMAGE_SIZE,
) -> tuple[np.ndarray, CropRect]:
    """Pad-square webcam pixels → model ``norm_crop`` space."""
    return webcam_pixels_to_norm_crop(
        kps_px,
        int(image_width),
        int(image_height),
        image_size=int(image_size),
    )


@dataclass
class TrackerSnapshot:
    t: float
    keypoints_px: np.ndarray | None
    keypoints_norm: np.ndarray | None
    image_wh: tuple[int, int]
    bridge: BridgeFrame | None
    rel: RelativePose | None
    fps: float
    faces: int
    reason: str = ""
    pts28: np.ndarray | None = None
    frame_shape: tuple[int, ...] | None = None
    skeleton_method: str = "none"
    body_lost: bool = False
    coord_space: str = COORD_NORM_CROP
    crop: CropRect | None = None
    image_size: int = LIVE_IMAGE_SIZE
    mirrored: bool = False
    # Mirrored BGR display frame (copied) for live camera diagnostic preview.
    frame_bgr: np.ndarray | None = None
    capture_ok: bool = True
    capture_health: str = "ok"
    read_fail_streak: int = 0
    capture_session: int = 0
    hair_segments_px: list | None = None
    hair_segments_norm: list | None = None
    hair_method: str = "none"
    hair_lost: bool = False


def is_body_tracked(method: str | None) -> bool:
    """True when body comes from a real pose detector (not synth / held)."""
    m = str(method or "").lower()
    if not m or m in ("none", "unknown"):
        return False
    if "synthetic" in m or m.endswith("_held") or m == "held":
        return False
    return any(tag in m for tag in ("mediapipe", "yolo", "dwpose", "tracked"))


def body_method_kind(method: str | None) -> str:
    """Classify skeleton_method into tracked | synthetic | held | none."""
    m = str(method or "").lower()
    if not m or m == "none":
        return "none"
    if m.endswith("_held") or m == "held":
        return "held"
    if "synthetic" in m:
        return "synthetic"
    if is_body_tracked(m):
        return "tracked"
    return "unknown"


def body_tracking_label(
    method: str | None,
    *,
    lost: bool = False,
    tracking: bool = True,
    drive_pose: bool = True,
    waiting: bool = False,
    visible_body: int | None = None,
    age: float | None = None,
) -> str:
    """Human-readable body-tracker health for the UI status line.

    States: OFF | WAITING | ACTIVE (human) | SYNTHETIC | HELD/LOST | NONE.
    """
    if not tracking:
        return "Body: OFF"
    if waiting:
        return "Body: WAITING (no face)"

    kind = body_method_kind(method)
    method_s = str(method or "none")
    vis = ""
    if visible_body is not None:
        vis = f"  joints={int(visible_body)}/7"
    age_s = ""
    if age is not None and age < 1e8:
        age_s = f"  age={float(age):.2f}s"
    drive = "" if drive_pose else "  drive=off"

    if lost or kind == "held":
        return f"Body: HELD/LOST — {method_s}{vis}{age_s}{drive}"
    if kind == "tracked":
        return f"Body: ACTIVE (human) — {method_s}{vis}{age_s}{drive}"
    if kind == "synthetic":
        return f"Body: SYNTHETIC (face fallback) — {method_s}{vis}{age_s}{drive}"
    if kind == "none":
        return f"Body: NONE (no upper body){vis}{age_s}{drive}"
    return f"Body: {kind.upper()} — {method_s}{vis}{age_s}{drive}"


def body_tracking_active(method: str | None, *, lost: bool = False) -> bool:
    """True only for live human body tracking (not synth / held / lost)."""
    return bool(is_body_tracked(method) and not lost)


class LivePoserTracker:
    """Background camera + OSF → (37,4) KEYPOINT_SCHEMA producer."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        model: int = 3,
        sensitivity: float = 50.0,
        smoothing: float = 40.0,
        use_gaze: bool = True,
        use_iris: bool = True,
        use_skeleton: bool = True,
        use_hair: bool = False,
        threads: int = 1,
        faces: int = 1,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.model = int(model)
        self.sensitivity = float(sensitivity)
        self.smoothing = float(smoothing)
        self.threads = int(threads)
        self.faces = int(faces)
        self.mirror = False

        self._face = FaceTracker()
        self._iris = IrisTracker(use_iris=use_iris, use_gaze=use_gaze)
        self._body = BodyTracker(enabled=use_skeleton)
        self._hair = HairTracker(enabled=use_hair)

        self.osf = resolve_openseeface(None)
        self.model_dir = str(self.osf / "models")

        self._stop = threading.Event()
        self._recover_capture = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.running = False
        self.camera_index: int | None = None
        self.camera_name: str = ""
        self.error: str | None = None

        self.calib = CenterCalibration()
        self._smoother = MotionSmoother()

        self._latest: TrackerSnapshot | None = None
        self._track_fps = 0.0
        self._capture_session = 0

    @property
    def use_iris(self) -> bool:
        return self._iris.use_iris

    @use_iris.setter
    def use_iris(self, value: bool) -> None:
        self._iris.use_iris = bool(value)

    @property
    def use_gaze(self) -> bool:
        return self._iris.use_gaze

    @use_gaze.setter
    def use_gaze(self, value: bool) -> None:
        self._iris.use_gaze = bool(value)

    @property
    def use_skeleton(self) -> bool:
        return self._body.enabled

    @use_skeleton.setter
    def use_skeleton(self, value: bool) -> None:
        self._body.enabled = bool(value)

    @property
    def use_hair(self) -> bool:
        return self._hair.enabled

    @use_hair.setter
    def use_hair(self, value: bool) -> None:
        self._hair.enabled = bool(value)

    @property
    def track_fps(self) -> float:
        return float(self._track_fps)

    def age_seconds(self) -> float:
        snap = self.latest_snapshot()
        if snap is None:
            return 1e9
        return max(0.0, time.time() - snap.t)

    def latest_snapshot(self) -> TrackerSnapshot | None:
        with self._lock:
            return self._latest

    def latest_keypoints(self, *, normalized: bool = True) -> np.ndarray | None:
        snap = self.latest_snapshot()
        if snap is None:
            return None
        if snap.t and (time.time() - snap.t) > FACE_MAX_AGE_S:
            return None
        return snap.keypoints_norm if normalized else snap.keypoints_px

    def request_capture_recovery(self) -> None:
        """Ask the tracking thread to reopen the current camera."""
        if self.running:
            self._recover_capture.set()

    def center(self) -> RelativePose | None:
        """Instant center calibration from latest face landmarks."""
        with self._lock:
            snap = self._latest
            pts28 = None if snap is None or snap.pts28 is None else snap.pts28.copy()
            shape = None if snap is None else snap.frame_shape
        if pts28 is None or shape is None:
            raise RuntimeError("No face tracked yet — look at the camera first.")
        self.calib.capture(pts28, shape, conf=1.0)
        self.calib.flash_until = time.time() + 0.45
        return self.calib.remap(pts28, shape)

    def reset_center(self) -> None:
        self.calib.clear()

    def set_mouth_osf_map(self, raw: object | None) -> dict[int, int]:
        """Replace overlay 20–27 ← OSF ids. Takes effect on the next frame."""
        return self._face.set_mouth_osf_map(raw)

    def mouth_osf_map(self) -> dict[int, int]:
        return self._face.mouth_osf_map()

    def start(self, camera_index: int, *, camera_name: str = "") -> None:
        # Always fully release the previous capture before reopening — DroidCam /
        # DirectShow reject a second open while the old reader still holds the device.
        if self._thread is not None and self._thread.is_alive():
            self.stop(settle=True)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=CAPTURE_STOP_JOIN_S)
            if self._thread.is_alive():
                raise RuntimeError(
                    "Previous LivePoser camera thread is still holding the device. "
                    "Wait a second and try Start LivePoser again."
                )
            self._thread = None

        self.error = None
        self.camera_index = int(camera_index)
        self.camera_name = camera_name or f"Camera {camera_index}"
        self._stop.clear()
        self._recover_capture.clear()
        self.running = True
        self._smoother.reset()
        self._body.reset()
        self._hair.reset()
        self.calib.clear()
        with self._lock:
            self._latest = None
        self._thread = threading.Thread(
            target=self._loop,
            args=(int(camera_index), self.camera_name),
            name="live-poser-track",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, settle: bool = True) -> None:
        self._stop.set()
        self.running = False
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=CAPTURE_STOP_JOIN_S)
        # Keep the handle if join timed out so start() can see the device is still held.
        if thread is None or not thread.is_alive():
            self._thread = None
        with self._lock:
            self._latest = None
        if settle:
            time.sleep(CAPTURE_RELEASE_SETTLE_S)

    def _publish_capture_heartbeat(
        self,
        *,
        health: str,
        reason: str,
        fail_streak: int,
    ) -> None:
        """Publish camera health without presenting stale keypoints as fresh."""
        with self._lock:
            prev = self._latest
            frame = None
            if prev is not None and prev.frame_bgr is not None:
                frame = np.ascontiguousarray(prev.frame_bgr.copy())
            image_wh = prev.image_wh if prev is not None else (self.width, self.height)
            frame_shape = prev.frame_shape if prev is not None else None
            mirrored = prev.mirrored if prev is not None else bool(self.mirror)
            self._latest = TrackerSnapshot(
                t=time.time(),
                keypoints_px=None,
                keypoints_norm=None,
                image_wh=image_wh,
                bridge=None,
                rel=None,
                fps=float(self._track_fps),
                faces=0,
                reason=reason,
                frame_shape=frame_shape,
                body_lost=True,
                mirrored=mirrored,
                frame_bgr=frame,
                capture_ok=False,
                capture_health=health,
                read_fail_streak=int(fail_streak),
                capture_session=int(self._capture_session),
            )

    def _filter_settings(self) -> FilterSettings:
        return FilterSettings(sensitivity=self.sensitivity, smoothing=self.smoothing)

    def _loop(self, camera_index: int, camera_name: str) -> None:
        capture = None
        ema = 0.0
        try:
            ensure_com()
            os.environ["OMP_NUM_THREADS"] = str(self.threads)
            ensure_osf_on_path(self.osf)

            # Always CPU — DiT/VAE own the GPU; CUDA iris races cause
            # "no face" / silent stalls after Stop stream → Start LivePoser.
            self._iris.load(device="cpu")
            self._body.load(device="cpu")
            # Hair stays off the live camera loop by default. Per-frame
            # animeseg on CPU was the tracking FPS cliff.
            self._hair.load(device="cpu")

            # Resolve CameraInfo for DirectShow name probing. Prefer name match so
            # index drift after Refresh / virtual-cam reopen doesn't open the wrong device.
            infos = list_camera_infos()
            cam = next((c for c in infos if c.index == camera_index), None)
            if cam is None and camera_name:
                cam = next(
                    (
                        c
                        for c in infos
                        if c.name.strip().lower() == camera_name.strip().lower()
                    ),
                    None,
                )
            if cam is None:
                cam = CameraInfo(index=camera_index, name=camera_name, backend="opencv")
            else:
                camera_index = int(cam.index)
                self.camera_index = camera_index

            def _open_capture() -> CameraCapture:
                return CameraCapture(
                    cam.index,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    backend="auto",
                    osf=self.osf,
                    camera_name=cam.name,
                )

            capture = None
            open_err: Exception | None = None
            for attempt in range(CAPTURE_OPEN_ATTEMPTS):
                if self._stop.is_set():
                    return
                try:
                    if attempt > 0:
                        delay = CAPTURE_REOPEN_BACKOFF_S[
                            min(attempt - 1, len(CAPTURE_REOPEN_BACKOFF_S) - 1)
                        ]
                        print(
                            f"Camera open retry {attempt + 1}/{CAPTURE_OPEN_ATTEMPTS} "
                            f"after {delay:.2f}s ({cam.name})"
                        )
                        if self._stop.wait(delay):
                            return
                    capture = _open_capture()
                    open_err = None
                    break
                except Exception as exc:
                    open_err = exc
                    print(f"Camera open failed (attempt {attempt + 1}): {exc}")
            if capture is None:
                raise RuntimeError(
                    str(open_err)
                    if open_err is not None
                    else f"Could not open camera {camera_index}"
                )
            self._capture_session += 1

            frame = None
            for _ in range(40):
                if self._stop.is_set():
                    return
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
            if frame is None:
                raise RuntimeError(
                    f"Camera opened ({capture.name} / {capture.backend}) but returned no frames."
                )

            health, _mean = frame_health(frame)
            if health == "black":
                print("Camera feed looks black - try another resolution / device")

            frame = fit_frame(frame, MAX_TRACK_WIDTH)
            h, w = frame.shape[:2]
            filters0 = self._filter_settings()
            self._tracker = build_tracker(
                w,
                h,
                self.model,
                self.faces,
                self.threads,
                filters0.detection_threshold(),
                filters0.landmark_threshold(),
                no_gaze=not self.use_gaze,
                model_dir=self.model_dir,
                osf=self.osf,
                try_hard=True,
                use_retinaface=1,
            )
            print(f"Tracking: {capture.name} [{capture.backend}] {w}x{h}")

            read_fail_streak = 0
            black_frame_streak = 0
            reopen_attempt = 0
            no_face_since: float | None = None
            last_detector_rebuild_t = 0.0
            detector_rebuilds = 0
            while not self._stop.is_set():
                recovery_requested = self._recover_capture.is_set()
                if recovery_requested:
                    self._recover_capture.clear()
                    read_fail_streak = CAPTURE_READ_FAIL_LIMIT - 1
                    ok, frame = False, None
                    capture_health = "reconnect_requested"
                else:
                    ok, frame = capture.read()
                    capture_health = str(getattr(capture, "last_health", "ok") or "ok")
                if not ok or frame is None or capture_health != "ok":
                    if capture_health == "black" and frame is not None:
                        black_frame_streak += 1
                        fail_count = black_frame_streak
                        limit = CAPTURE_BLACK_FRAME_LIMIT
                        reason = "Camera frames are black/flat"
                    else:
                        read_fail_streak += 1
                        fail_count = read_fail_streak
                        limit = CAPTURE_READ_FAIL_LIMIT
                        reason = "Camera frame stalled"
                    self._publish_capture_heartbeat(
                        health=capture_health if capture_health != "ok" else "stalled",
                        reason=reason,
                        fail_streak=fail_count,
                    )
                    if fail_count < limit:
                        time.sleep(0.03)
                        continue

                    # Reopen in this same worker so Tk never touches camera APIs.
                    self._publish_capture_heartbeat(
                        health="reconnecting",
                        reason="Camera stalled — reconnecting",
                        fail_streak=fail_count,
                    )
                    try:
                        capture.release()
                    except Exception:
                        pass
                    delay = CAPTURE_REOPEN_BACKOFF_S[
                        min(reopen_attempt, len(CAPTURE_REOPEN_BACKOFF_S) - 1)
                    ]
                    if self._stop.wait(delay):
                        break
                    try:
                        capture = _open_capture()
                        self._capture_session += 1
                        self._smoother.reset()
                        self._body.reset()
                        self._hair.reset()
                        self.calib.clear()
                        read_fail_streak = 0
                        black_frame_streak = 0
                        reopen_attempt = 0
                        print(
                            f"Camera recovered: {capture.name} "
                            f"[{capture.backend}] session={self._capture_session}"
                        )
                    except Exception as exc:
                        reopen_attempt += 1
                        self._publish_capture_heartbeat(
                            health="reconnecting",
                            reason=f"Camera reconnect failed: {exc}",
                            fail_streak=fail_count,
                        )
                    continue

                read_fail_streak = 0
                black_frame_streak = 0
                reopen_attempt = 0
                if frame is None:
                    time.sleep(0.01)
                    continue
                frame = fit_frame(frame, MAX_TRACK_WIDTH)
                if frame.shape[0] != h or frame.shape[1] != w:
                    import cv2

                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                t0 = time.perf_counter()
                faces = self._tracker.predict(frame)
                dt = time.perf_counter() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                ema = inst if ema == 0 else (ema * 0.85 + inst * 0.15)
                self._track_fps = ema

                filters = self._filter_settings()
                face_thr = filters.detection_threshold()
                alive = [
                    f
                    for f in faces
                    if getattr(f, "alive", False)
                    and f.lms is not None
                    and float(getattr(f, "conf", 1.0) or 0.0) >= face_thr * 0.5
                ]
                now_wall = time.monotonic()
                frame_status, _mean = frame_health(frame)
                if alive:
                    no_face_since = None
                else:
                    if no_face_since is None:
                        no_face_since = now_wall
                    no_face_for = now_wall - no_face_since
                    # Real camera content but no face → rebuild OSF sooner; reopen if stuck.
                    if (
                        frame_status == "ok"
                        and no_face_for >= NO_FACE_REBUILD_S
                        and detector_rebuilds < NO_FACE_REBUILD_MAX
                        and now_wall - last_detector_rebuild_t >= NO_FACE_REBUILD_COOLDOWN_S
                    ):
                        try:
                            close_tracker(self._tracker)
                            self._tracker = build_tracker(
                                w,
                                h,
                                self.model,
                                self.faces,
                                self.threads,
                                filters.detection_threshold(),
                                filters.landmark_threshold(),
                                no_gaze=not self.use_gaze,
                                model_dir=self.model_dir,
                                osf=self.osf,
                                try_hard=True,
                                use_retinaface=1,
                            )
                            self._smoother.reset()
                            last_detector_rebuild_t = now_wall
                            detector_rebuilds += 1
                            print(
                                f"Face detector rebuilt after {no_face_for:.1f}s with no face "
                                f"(frame={frame_status}, rebuild={detector_rebuilds}/{NO_FACE_REBUILD_MAX})"
                            )
                            # Immediate retry on this frame.
                            faces = self._tracker.predict(frame)
                            alive = [
                                f
                                for f in faces
                                if getattr(f, "alive", False)
                                and f.lms is not None
                                and float(getattr(f, "conf", 1.0) or 0.0)
                                >= face_thr * 0.5
                            ]
                            if alive:
                                no_face_since = None
                        except Exception as exc:
                            print(f"Face detector rescan failed: {exc}")
                    if (
                        frame_status == "ok"
                        and no_face_since is not None
                        and (now_wall - no_face_since) >= NO_FACE_REOPEN_S
                    ):
                        # Force the capture reopen path (same as stall recovery).
                        print(
                            f"No face for {now_wall - no_face_since:.1f}s with healthy frames "
                            "— reopening camera"
                        )
                        self._recover_capture.set()
                        no_face_since = now_wall
                        continue
                mirror = bool(self.mirror)
                rel = None
                bridge = None
                reason = ""
                face_obj = alive[0] if alive else None
                face_track = (
                    self._face.decode(
                        face_obj, landmark_threshold=filters.landmark_threshold()
                    )
                    if face_obj is not None
                    else FaceTrack()
                )
                iris_track = self._iris.detect(
                    frame, face_obj, face_track.pts28, filters
                )

                raw_body = None
                raw_method = "none"
                if self.use_skeleton:
                    raw_body, raw_method = self._body.resolve(
                        frame,
                        None,
                        filters,
                        allow_synth_fallback=not self._body.has_model,
                    )
                    if raw_body is None and not self._body.has_hold and face_track.alive:
                        # Retry with face landmarks; keep the real tracker if present
                        # so MediaPipe/YOLO get another chance before synth fallback.
                        raw_body, raw_method = self._body.resolve(
                            frame,
                            face_track.pts28,
                            filters,
                            allow_synth_fallback=True,
                        )
                body_track = (
                    self._body.hold_update(raw_body, raw_method)
                    if self.use_skeleton
                    else BodyTrack()
                )
                hair_track = self._hair.detect(frame)
                if not alive:
                    reason = detection_reason(frame, alive) or "No face in view"

                pts28, eye_lower, right_iris, left_iris, body7 = self._smoother.apply(
                    pts28=face_track.pts28,
                    eye_lower=face_track.eye_lower,
                    right_iris=iris_track.right,
                    left_iris=iris_track.left,
                    body7=body_track.joints,
                    smoothing=filters.smoothing,
                    freeze_body=body_track.lost,
                )
                face_track = FaceTrack(
                    pts28=pts28,
                    eye_lower=eye_lower,
                    mouth_osf=face_track.mouth_osf,
                    alive=pts28 is not None,
                )
                iris_track = IrisTrack(
                    right=right_iris,
                    left=left_iris,
                    method=iris_track.method,
                )
                body_track = BodyTrack(
                    joints=body7,
                    method=body_track.method,
                    lost=body_track.lost,
                )

                import cv2

                display_frame = frame
                display_shape = frame.shape
                if mirror:
                    display_frame = cv2.flip(frame, 1)
                    display_shape = display_frame.shape
                    width = display_shape[1]
                    if face_track.pts28 is not None:
                        face_track = FaceTracker.flip(face_track, width)
                        iris_track = IrisTracker.flip(iris_track, width)
                    if body_track.joints is not None:
                        body_track = BodyTracker.flip(body_track, width)
                    if hair_track.segments:
                        hair_track = HairTracker.flip(hair_track, width)

                kps_px = None
                kps_norm = None
                crop = None
                hair_segments_px = None
                hair_segments_norm = None
                merged = None
                if alive and face_track.pts28 is not None:
                    rel = self.calib.remap(face_track.pts28, display_shape)
                    pose_dict = self.calib.to_dict(rel)
                    if (
                        self.use_skeleton
                        and body_track.method == "synthetic_from_face"
                        and not body_track.lost
                    ):
                        # Re-synth with calibrated head angles (better than the
                        # zero-angle fallback produced before CenterCalibration).
                        body_track = self._body.resynth_from_face(
                            face_track.pts28,
                            pitch=rel.pitch,
                            yaw=rel.yaw,
                            roll=rel.roll,
                        )
                    merged = merge_tracks(
                        image_shape=display_shape,
                        face=face_track,
                        iris=iris_track,
                        body=body_track,
                        hair=hair_track,
                        mirrored=mirror,
                        pose=pose_dict,
                        meta={
                            "fps": self._track_fps,
                            "osf_model": self.model,
                            "sensitivity": filters.sensitivity,
                            "smoothing": filters.smoothing,
                            "det_thr": filters.detection_threshold(),
                            "body_lost": body_track.lost,
                        },
                        image_size=LIVE_IMAGE_SIZE,
                        calibrated=bool(rel is not None and rel.calibrated),
                    )
                elif body_track.joints is not None:
                    body_track = BodyTrack(
                        joints=body_track.joints,
                        method=body_track.method,
                        lost=True,
                    )
                    merged = merge_tracks(
                        image_shape=display_shape,
                        face=None,
                        iris=iris_track,
                        body=body_track,
                        hair=hair_track,
                        mirrored=mirror,
                        meta={"fps": self._track_fps, "body_lost": True},
                        image_size=LIVE_IMAGE_SIZE,
                        calibrated=False,
                    )
                    reason = "face lost — skeleton held"
                if merged is not None:
                    bridge = merged.bridge
                    kps_px = merged.keypoints_px
                    kps_norm = merged.keypoints_norm
                    crop = merged.crop
                    hair_segments_px = merged.hair_segments_px
                    hair_segments_norm = merged.hair_segments_norm

                # Defensive copy so the UI thread can safely read while the
                # capture loop overwrites its working buffer next iteration.
                frame_bgr = None
                if display_frame is not None:
                    preview = np.ascontiguousarray(display_frame.copy())
                    HairTracker.draw(preview, hair_track)
                    frame_bgr = preview

                snap = TrackerSnapshot(
                    t=time.time(),
                    keypoints_px=None if kps_px is None else kps_px.copy(),
                    keypoints_norm=None if kps_norm is None else kps_norm.copy(),
                    image_wh=(int(display_shape[1]), int(display_shape[0])),
                    bridge=bridge,
                    rel=rel,
                    fps=float(self._track_fps),
                    faces=len(alive),
                    reason=reason,
                    pts28=None if face_track.pts28 is None else face_track.pts28.copy(),
                    frame_shape=tuple(int(x) for x in display_shape),
                    skeleton_method=str(body_track.method or "none"),
                    body_lost=bool(body_track.lost),
                    coord_space=COORD_NORM_CROP if kps_norm is not None else "",
                    crop=crop if kps_norm is not None else None,
                    image_size=LIVE_IMAGE_SIZE,
                    mirrored=bool(mirror),
                    frame_bgr=frame_bgr,
                    capture_ok=True,
                    capture_health="ok",
                    read_fail_streak=0,
                    capture_session=int(self._capture_session),
                    hair_segments_px=hair_segments_px,
                    hair_segments_norm=hair_segments_norm,
                    hair_method=str(hair_track.method or "none"),
                    hair_lost=bool(hair_track.lost),
                )
                with self._lock:
                    self._latest = snap

        except Exception as exc:
            self.error = str(exc)
            print(f"LivePoserTracker error: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            try:
                close_tracker(self._tracker)
            except Exception:
                pass
            self._tracker = None
            self._iris.close()
            self._body.close()
            self._hair.close()
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass
            try:
                uninit_com()
            except Exception:
                pass
