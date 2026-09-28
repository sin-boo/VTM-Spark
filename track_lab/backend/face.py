"""Anime-mesh bench. Live tracking comes from OSF camera or iFacialMocap."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np

from .anime import AnimeMeshError, draw_label28, fit_mesh, reset_anime_mesh, rest_too_small
from . import debug_log
from .cameras import list_cameras, load_camera_choice, pick_default, save_camera_index


def _same_index(cam: dict[str, object], index: int) -> bool:
    try:
        return int(cam["index"]) == int(index)
    except (KeyError, TypeError, ValueError):
        return False


def _camera_name(cameras: list[dict[str, object]], index: int) -> str:
    for cam in cameras:
        if _same_index(cam, index):
            return str(cam.get("name") or "").strip()
    return ""
from .calibrate import calibrator
from .feel import feel
from .hair import HAIR_CLASSES, build_hair_rig, detect_hair, follow_hair, rig_rest_hair
from .eye_bits import bits as eye_bits
from .mouth_bits import bits as mouth_bits
from .ifm import DEFAULT_PORT, drive_ifm, look_quiet
from .ifm_cam import IfmCam
from .iris import payload_to_hits, raw_debug, rest_look_from_cam
from .iris import retarget as retarget_iris
from .iris import track_still
from .offsets import apply_points as offset_points
from .offsets import apply_rows as offset_rows
from .offsets import clear as clear_offsets
from .offsets import dump as dump_offsets
from .offsets import nudge as nudge_offset
from .offsets import parse as parse_offsets
from .osf_cam import OsfCam, OsfFrame
from .skeleton import follow_skeleton, skeleton_from_still
from .travel_box import (
    apply_limits,
    lab_feel_caps,
    limiter_rects_px,
    pack_overlay,
    travel,
    unpack_face,
    unpack_iris,
    unpack_skeleton,
)
from harness.hub import hub
from harness.pack import frame_from_bench, status_from_bench

from .presets import (
    MOUTH_SLOTS,
    book,
    empty_weights,
    pts_to_json,
)
from .record import MovementRecorder
from .retarget import FaceExpr
from .rig import FaceRig
from .sides import ifm_canonical, ifm_look_canonical, selfie_of, to_screen

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
SOURCE_NAME = "source.png"
PREVIEW_MAX = 960
PARTS_PATH = OUTPUT_DIR / "overlay_parts.json"
IFM_PATH = OUTPUT_DIR / "ifm.json"


def source_path() -> Path:
    preferred = INPUT_DIR / SOURCE_NAME
    if preferred.is_file():
        return preferred
    for path in sorted(INPUT_DIR.iterdir()) if INPUT_DIR.is_dir() else []:
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return path
    return preferred


def _read_bgr(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _decode_bgr(payload: bytes) -> np.ndarray | None:
    if not payload:
        return None
    data = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def same_still_bgr(cur: np.ndarray | None, nxt: np.ndarray | None) -> bool:
    """True when VTM re-pushed the same character still. Keep authored shapes."""
    if cur is None or nxt is None:
        return False
    if cur.shape != nxt.shape:
        return False
    return bool(np.array_equal(cur, nxt))


def _fit_preview(bgr: np.ndarray, max_side: int = PREVIEW_MAX) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    return cv2.resize(
        bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _encode_jpeg(bgr: np.ndarray) -> bytes:
    preview = _fit_preview(bgr)
    ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Could not encode frame")
    return bytes(buf)


@dataclass
class FaceBench:
    source_bgr: np.ndarray | None = None
    overlay_bgr: np.ndarray | None = None
    source_mtime: float = 0.0
    last_ms: float = 0.0
    last_faces: int = 0
    last_error: str = ""
    generation: int = 0
    last_tracker: str = "anime"
    rest_pts: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._osf = OsfCam()
        self._ifm = IfmCam()
        self._source = "camera"
        self._mirror = False
        self._rig = FaceRig()
        self._expr = FaceExpr()
        self._live_pts: np.ndarray | None = None
        self._smooth_pts: np.ndarray | None = None
        self._snap_smooth = False
        self._finishing = False
        self._live_pose: tuple[str, object, np.ndarray | None, np.ndarray | None] | None = None
        self._smooth_hair: list[dict[str, object]] = []
        self._smooth_skeleton: list[dict[str, object]] = []
        self._iris: list[dict[str, object]] = []
        self._smooth_iris: list[dict[str, object]] = []
        self._iris_method = "none"
        self._iris_rest: list[dict[str, object]] = []
        self._iris_rest_method = "none"
        self._iris_cam: list[dict[str, object]] = []
        self._look: dict[str, float] | None = None
        self._look_rest: dict[str, object] = {}
        self._point_offsets: dict[int, tuple[float, float]] = {}
        self._mouth_box: list[float] | None = None
        self._mouth_cage: list[float] | None = None
        self._hair: list[dict[str, object]] = []
        self._hair_rig = None
        self._skeleton: list[dict[str, object]] = []
        self._skeleton_rest: list[dict[str, object]] = []
        self._weights = empty_weights()
        self._head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        self._blink = {"l": 0.0, "r": 0.0}
        self.camera_bgr: bytes | None = None
        self._cameras: list[dict[str, object]] | None = None
        self.camera_index = 0
        self._saved_camera, self._saved_camera_name = load_camera_choice()
        if self._saved_camera is not None:
            self.camera_index = self._saved_camera
        if self.rest_pts is None:
            self.rest_pts = book.template(None)
        if self.rest_pts is not None:
            self.last_faces = 1
        self._recorder = MovementRecorder()
        self._load_parts()
        self._load_ifm()
        self._apply_mirror(self._mirror)
        # Feel caps follow the persisted travel box (same as set_travel).
        feel.update(lab_feel_caps(travel.payload()))
        self._ensure_source()

    def _refresh_cameras(self) -> None:
        """Re-list devices and re-find the chosen camera by name."""
        try:
            self._cameras = list_cameras()
        except Exception:
            self._cameras = []
        self.camera_index = pick_default(
            self._cameras,
            self._saved_camera,
            saved_name=self._saved_camera_name,
        )

    def refresh_cameras(self) -> dict[str, object]:
        if not self._osf.running:
            self._refresh_cameras()
        return self.status(publish=True)

    def _camera_fields(self) -> dict[str, object]:
        # None = never listed. An empty list is cached too, so a machine with
        # no camera does not re-probe on every status.
        if self._cameras is None:
            self._refresh_cameras()
        return {
            "camera_index": int(self.camera_index),
            "cameras": self._cameras or [],
        }

    def _live_source(self) -> str:
        if self._ifm.running:
            return "ifm"
        if self._osf.running:
            return "camera"
        return self._source

    def _live_fields(self) -> dict[str, object]:
        live = self._osf.running or self._ifm.running
        tracker = "ifm" if self._ifm.running else "osf" if self._osf.running else self.last_tracker
        with self._lock:
            pts = self._live_pts if live else None
            box = list(self._mouth_box) if live and self._mouth_box else []
            cage = list(self._mouth_cage) if live and self._mouth_cage else []
            hair = list(self._hair)
            skeleton = list(self._skeleton)
            iris = list(self._iris)
            iris_method = str(self._iris_method)
            debug = raw_debug(self._iris_cam, self._look)
            offsets = dict(self._point_offsets)
            weights = {key: round(float(value), 3) for key, value in self._weights.items()}
            head = dict(self._head)
            blink = dict(self._blink)
        mixed = pts_to_json(pts) if pts is not None else None
        if mixed:
            mixed = offset_points(mixed, offsets)
        iris = offset_rows(iris, offsets)
        skeleton = offset_rows(skeleton, offsets)
        return {
            "live": live,
            "tracker": tracker,
            "weights": weights,
            "head": head,
            "blink": blink,
            "live_points": mixed or [],
            "calib": calibrator.payload(),
            "mouth_box": box,
            "mouth_cage": cage,
            "hair": hair,
            "hair_method": "lab_follow" if hair else "none",
            "skeleton": skeleton,
            "iris": iris,
            "iris_method": iris_method,
            "iris_cam": debug["iris_cam"],
            "look": debug["look"],
            "point_offsets": dump_offsets(offsets),
            "travel_box": travel.payload(),
            "travel_rects": self._travel_rects(),
            **self._recorder.payload(),
            **mouth_bits.payload(),
            **eye_bits.payload(),
        }

    def _rest_overlay(self) -> np.ndarray | None:
        if self.rest_pts is None:
            return None
        return pack_overlay(self.rest_pts, self._skeleton_rest, self._iris_rest)

    def _travel_rects(self) -> dict[str, object]:
        return limiter_rects_px(self._rest_overlay(), travel.payload())

    def _apply_travel_limits(
        self,
        posed: np.ndarray | None,
        skeleton: list[dict[str, object]],
        iris: list[dict[str, object]],
        hair: list[dict[str, object]],
    ) -> tuple[
        np.ndarray | None,
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        if posed is None:
            return posed, skeleton, iris, hair
        rest = self._rest_overlay()
        if rest is None:
            return posed, skeleton, iris, hair
        live = pack_overlay(posed, skeleton, iris)
        limited, hair_out = apply_limits(live, rest, travel.payload(), hair=hair)
        return (
            unpack_face(limited, posed),
            unpack_skeleton(limited, skeleton),
            unpack_iris(limited, iris),
            list(hair_out) if hair_out is not None else [],
        )

    def _publish(self, payload: dict[str, object] | None = None, *, frame: bool = False) -> None:
        try:
            if payload is not None:
                hub.publish(status_from_bench(self, payload, clients=hub.clients))
            if frame or payload is not None:
                hub.publish(frame_from_bench(self, clients=hub.clients))
        except Exception:
            return

    def status(self, *, publish: bool = False) -> dict[str, object]:
        self._ensure_source()
        src = source_path()
        h = int(self.source_bgr.shape[0]) if self.source_bgr is not None else 0
        w = int(self.source_bgr.shape[1]) if self.source_bgr is not None else 0
        live = self._live_fields()
        payload = book.payload(self.rest_pts)
        if live["live"] and live["live_points"]:
            payload["points"] = live["live_points"]
        elif payload.get("points"):
            payload["points"] = offset_points(
                payload["points"], parse_offsets(live.get("point_offsets"))
            )
        out = {
            "ok": True,
            "ready": self.rest_pts is not None or book.template(None) is not None,
            "has_source": src.is_file(),
            "source_path": str(src) if src.is_file() else "",
            "width": w,
            "height": h,
            "tracker": live["tracker"],
            "faces": self.last_faces,
            "ms": round(self.last_ms, 2),
            "generation": self.generation,
            "error": self.last_error,
            **payload,
            **live,
            **self._camera_fields(),
            "feel": feel.payload(),
            "travel_box": travel.payload(),
            "travel_rects": self._travel_rects(),
            "source": self._live_source(),
            "ifm": self._ifm.payload(),
            "mirror": bool(self._mirror),
        }
        if publish:
            self._publish(out)
        return out

    def live_status(self) -> dict[str, object]:
        live = self._live_fields()
        fallback = book.current()
        if fallback is None:
            fallback = self.rest_pts
        return {
            "ok": True,
            "live": live["live"],
            "tracker": live["tracker"],
            "faces": self.last_faces,
            "ms": round(self.last_ms, 2),
            "error": self.last_error,
            # While live, never substitute the authored preset: the client
            # would snap the character to rest for a frame. Empty = hold.
            "points": live["live_points"]
            if live["live"]
            else offset_points(
                pts_to_json(fallback), parse_offsets(live.get("point_offsets"))
            ),
            "weights": live["weights"],
            "head": live["head"],
            "turn": self._rig.debug(),
            "blink": live["blink"],
            "camera_index": int(self.camera_index),
            "source": self._live_source(),
            "ifm": self._ifm.payload(),
            "calib": live["calib"],
            "mouth_points": live.get("mouth_points") or [],
            "eye_points": live.get("eye_points") or [],
            "mouth_box": live.get("mouth_box") or [],
            "mouth_cage": live.get("mouth_cage") or [],
            "hair": live.get("hair") or [],
            "hair_method": live.get("hair_method") or ("lab_follow" if live.get("hair") else "none"),
            "skeleton": live.get("skeleton") or [],
            "iris": live.get("iris") or [],
            "iris_method": live.get("iris_method") or "none",
            "iris_cam": live.get("iris_cam") or [],
            "look": live.get("look"),
            "point_offsets": live.get("point_offsets") or [],
            "feel": feel.payload(),
            "travel_box": travel.payload(),
            "travel_rects": self._travel_rects(),
            **self._recorder.payload(),
        }

    def source_jpeg(self) -> bytes | None:
        self._ensure_source()
        if self.source_bgr is None:
            return None
        return _encode_jpeg(self.source_bgr)

    def overlay_jpeg(self) -> bytes | None:
        if self.overlay_bgr is None:
            return None
        return _encode_jpeg(self.overlay_bgr)

    def camera_jpeg(self) -> bytes | None:
        with self._lock:
            payload = self.camera_bgr
        return payload or None

    def set_source(self, payload: bytes, filename: str = SOURCE_NAME) -> dict[str, object]:
        self.stop_live()
        image = _decode_bgr(payload)
        if image is None:
            self.last_error = f"Could not read {filename}"
            return self.status(publish=True)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = INPUT_DIR / SOURCE_NAME
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            self.last_error = "Could not save source.png"
            return self.status(publish=True)
        keep = same_still_bgr(self.source_bgr, image)
        dest.write_bytes(bytes(buf))
        self.source_bgr = image
        self.source_mtime = dest.stat().st_mtime
        self.last_error = ""
        self.last_tracker = "anime"
        self.generation += 1
        if keep:
            # Hair was cleaned when detected, or painted on the desk. Cleaning it
            # again carves green props out and erodes a pixel every re-push.
            if self.rest_pts is not None:
                self._paint(self.rest_pts)
            return self.status(publish=True)
        self.overlay_bgr = None
        self.rest_pts = None
        book.clear()
        self._hair = []
        self._hair_rig = None
        self._skeleton = []
        self._skeleton_rest = []
        self._point_offsets = {}
        self.last_faces = 0
        self.last_ms = 0.0
        return self.status(publish=True)

    def reset(self) -> dict[str, object]:
        self.stop_live()
        reset_anime_mesh()
        self.overlay_bgr = None
        self.rest_pts = book.template(None)
        self._hair = []
        self._hair_rig = None
        self._skeleton = []
        self._skeleton_rest = []
        self._point_offsets = {}
        self.last_faces = 1 if self.rest_pts is not None else 0
        self.last_ms = 0.0
        self.last_error = ""
        self.last_tracker = "anime"
        self.generation += 1
        if self.rest_pts is not None:
            self._paint(self.rest_pts)
        return self.status(publish=True)

    def track(self) -> dict[str, object]:
        self.stop_live()
        self._ensure_source(force=True)
        if self.source_bgr is None:
            self.last_error = f"Put a photo at {INPUT_DIR / SOURCE_NAME}"
            return self.status(publish=True)
        try:
            frame = self.source_bgr
            started = time.perf_counter()
            pts, _box = fit_mesh(frame)
            book.rebase(pts)
            self.last_ms = (time.perf_counter() - started) * 1000.0
            self.rest_pts = pts
            # Nudges were made on the old rest. Kept, they ride the new mesh
            # and a character created from this still saves them in.
            self._point_offsets = {}
            self.last_faces = 1
            self.last_tracker = "anime"
            # Four models on the still. They overlay together; none punches another.
            self._apply_still_iris(frame, pts)
            self._capture_parts(frame, pts)
            vis = draw_label28(frame, pts)
            self.overlay_bgr = vis
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUTPUT_DIR / "overlay.jpg"), vis)
            self.last_error = ""
        except AnimeMeshError as exc:
            self.last_error = str(exc)
            self.last_faces = 0
            self.rest_pts = None
            self._hair = []
            self._hair_rig = None
            self._skeleton = []
            self._skeleton_rest = []
            self._set_iris_rest([], "none")
            self.overlay_bgr = None
        except Exception as exc:
            self.last_error = str(exc)
            self.last_faces = 0
            self.rest_pts = None
            self._hair = []
            self._hair_rig = None
            self._skeleton = []
            self._skeleton_rest = []
            self._set_iris_rest([], "none")
            self.overlay_bgr = None
        return self.status(publish=True)

    def set_rest(self, body: dict[str, object]) -> dict[str, object]:
        """Install a character's packaged rest mesh on the still, no detection.

        A .vtm already carries the face mesh, iris and skeleton (and usually
        hair), so loading one must not re-run the face / landmark / iris /
        skeleton models. Hair is detected only when the pack has none yet.
        body: points (28 x [x, y, score]), iris ([{id, x, y, score, visible}]),
        skeleton ([{id, name?, x, y, score?}]), hair? ([{class, polygon}]);
        all in character pixels.
        """
        self.stop_live()
        self._ensure_source(force=True)
        if self.source_bgr is None:
            self.last_error = f"Put a photo at {INPUT_DIR / SOURCE_NAME}"
            return self.status(publish=True)
        raw = np.asarray(body.get("points") or [], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != 28 or raw.shape[1] < 2:
            self.last_error = "set_rest needs 28 face points"
            return self.status(publish=True)
        pts = np.ones((28, 3), dtype=np.float32)
        pts[:, : min(3, raw.shape[1])] = raw[:, :3]
        frame = self.source_bgr
        started = time.perf_counter()
        self.last_error = ""
        book.rebase(pts)
        self.rest_pts = pts
        # The pack's mesh already holds every edit; a nudge on top doubles it.
        self._point_offsets = {}
        self.last_faces = 1
        self.last_tracker = "anime"
        iris = [dict(row) for row in body.get("iris") or [] if isinstance(row, dict)]
        self._set_iris_rest(iris, "iris_pose" if iris else "none")
        hair = body.get("hair")
        if isinstance(hair, list) and hair:
            segs = [dict(seg) for seg in hair if isinstance(seg, dict)]
        else:
            try:
                segs = detect_hair(frame)
            except Exception as exc:
                segs = []
                self.last_error = f"Hair: {exc}"
        self._hair = segs
        self._hair_rig = build_hair_rig(segs, pts)
        skeleton = [dict(j) for j in body.get("skeleton") or [] if isinstance(j, dict)]
        if not skeleton:
            skeleton = skeleton_from_still(pts, frame)
        self._skeleton = skeleton
        self._skeleton_rest = [dict(j) for j in skeleton]
        self._save_parts()
        self.last_ms = (time.perf_counter() - started) * 1000.0
        vis = draw_label28(frame, pts)
        self.overlay_bgr = vis
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / "overlay.jpg"), vis)
        return self.status(publish=True)

    def apply_preset(self, name: str) -> dict[str, object]:
        if self._osf.running:
            self.last_error = "Stop OSF before editing mouth shapes"
            return self.status(publish=True)
        try:
            pts = book.apply(name, self.rest_pts)
        except ValueError as exc:
            self.last_error = str(exc)
            return self.status(publish=True)
        self.last_error = ""
        self._paint(pts)
        return self._authored_status(pts)

    def set_mouth(self, name: str, mouth: object) -> dict[str, object]:
        if self._osf.running:
            self.last_error = "Stop OSF before editing mouth shapes"
            return self.status(publish=True)
        try:
            pts = book.set_mouth(name, mouth, self.rest_pts)
        except ValueError as exc:
            self.last_error = str(exc)
            return self.status(publish=True)
        if name == "rest":
            self.rest_pts = pts
        with self._lock:
            for slot in MOUTH_SLOTS:
                self._point_offsets = clear_offsets(self._point_offsets, slot)
        self._save_parts()
        self.last_error = ""
        self._paint(pts)
        return self._authored_status(pts)

    def move_key(self, name: str, t: object) -> dict[str, object]:
        if self._osf.running:
            self.last_error = "Stop OSF before editing mouth shapes"
            return self.status(publish=True)
        try:
            book.move_key(name, t)
        except ValueError as exc:
            self.last_error = str(exc)
            return self.status(publish=True)
        self.last_error = ""
        return self.status(publish=True)

    def drop_key(self, name: str) -> dict[str, object]:
        if self._osf.running:
            self.last_error = "Stop OSF before editing mouth shapes"
            return self.status(publish=True)
        try:
            book.drop_key(name)
        except ValueError as exc:
            self.last_error = str(exc)
            return self.status(publish=True)
        self.last_error = ""
        return self.status(publish=True)

    def start_live(
        self,
        camera: int | None = None,
        source: str | None = None,
        host: str | None = None,
        port: int | None = None,
        mirror: object = None,
    ) -> dict[str, object]:
        self._ensure_source()
        rest = book.template(self.rest_pts)
        src = self.source_bgr
        small = False
        if rest is not None and src is not None:
            small = rest_too_small(rest, int(src.shape[1]), int(src.shape[0]))
        if rest is None or small:
            fitted = self.track()
            if str(fitted.get("error") or ""):
                return fitted
            rest = book.template(self.rest_pts)
        if rest is None:
            self.last_error = "Track a face first so rest exists"
            return self.status(publish=True)
        self.rest_pts = rest
        self._hair = []
        self._hair_rig = None
        if not self._hair or not self._skeleton_rest:
            self._load_parts()
        if self.source_bgr is not None and (not self._hair or not self._skeleton_rest):
            self._capture_parts(self.source_bgr, rest)
        if source:
            self.set_source_mode(str(source))
        if mirror is not None:
            self._apply_mirror(bool(mirror))
        if host is not None or port is not None:
            self.set_ifm(host, port)
        kind = self._source
        if kind == "ifm":
            if self._osf.running:
                self._osf.stop()
            if self._ifm.running:
                return self.status(publish=True)
            self._rig.reset()
            self._expr.reset()
            try:
                self._ifm.start(self._on_osf, host=self._ifm.host, port=self._ifm.port)
            except Exception as exc:
                self.last_error = str(exc)
                return self.status(publish=True)
            self.last_error = ""
            self.last_tracker = "ifm"
            return self.status(publish=True)
        if self._ifm.running:
            self._ifm.stop()
        if self._osf.running:
            self.last_error = ""
            return self.status(publish=True)
        self._rig.reset()
        self._expr.reset()
        if camera is not None:
            self.set_camera(int(camera))
            if self.last_error:
                return self.status(publish=True)
        # Indices shift when a device comes or goes; re-find the pick by name.
        self._refresh_cameras()
        try:
            self._osf.start(self._on_osf, index=int(self.camera_index))
        except Exception as exc:
            self.last_error = str(exc)
            return self.status(publish=True)
        self.last_error = ""
        self.last_tracker = "osf"
        return self.status(publish=True)

    def set_camera(self, index: int) -> dict[str, object]:
        if self._osf.running:
            self.last_error = "Stop OSF before changing cameras"
            return self.status(publish=True)
        if self._cameras is None:
            self._refresh_cameras()
        cameras = self._cameras or []
        # An empty list means enumeration failed; let OpenCV try the index.
        if cameras and not any(_same_index(cam, index) for cam in cameras):
            self.last_error = f"No camera at index {int(index)}"
            return self.status(publish=True)
        self.camera_index = int(index)
        self._saved_camera = self.camera_index
        name = _camera_name(cameras, self.camera_index)
        self._saved_camera_name = name
        save_camera_index(self.camera_index, name)
        self.last_error = ""
        return self.status(publish=True)

    @property
    def selfie(self) -> bool:
        """Mirror OFF: the character is your reflection (person-left on
        screen-left). Mirror ON: anatomical copy. See sides.py."""
        return selfie_of(self._mirror)

    def _apply_mirror(self, on: bool) -> None:
        """The one left/right switch. No rest recapture: L/R rows swap and
        X negates, so a held look-left becomes look-right at once."""
        self._mirror = bool(on)
        selfie = self.selfie
        self._expr.set_selfie(selfie)
        self._rig.selfie = selfie
        self._osf.preview_flip = selfie

    def set_mirror(self, on: bool) -> dict[str, object]:
        changed = bool(on) != self._mirror
        self._apply_mirror(on)
        if changed:
            self._save_ifm()
        self.last_error = ""
        return self.status(publish=True)

    def set_source_mode(self, source: str) -> dict[str, object]:
        kind = "ifm" if str(source).strip().lower() == "ifm" else "camera"
        if kind != self._live_source() and (self._osf.running or self._ifm.running):
            self.stop_live()
        self._source = kind
        self._save_ifm()
        self.last_error = ""
        return self.status(publish=True)

    def set_ifm(self, host: str | None = None, port: int | None = None) -> dict[str, object]:
        if (
            self._ifm.running
            and port is not None
            and int(port) > 0
            and int(port) != int(self._ifm.port)
        ):
            self.last_error = "Stop listening before changing the port"
            return self.status(publish=True)
        # An empty host clears the saved phone IP (find-the-phone mode).
        self._ifm.configure(host, port)
        self._save_ifm()
        self.last_error = ""
        return self.status(publish=True)

    def set_feel(self, body: object) -> dict[str, object]:
        feel.update(body)
        self.last_error = ""
        return self.status(publish=True)

    def set_travel(self, body: object) -> dict[str, object]:
        box, changed = travel.update(body if isinstance(body, dict) else {})
        if changed:
            feel.update(lab_feel_caps(box))
            # Held head pose: re-clip now, so the slider moves the overlay
            # without waiting for the next camera sample to ease in.
            self._snap_smooth = True
            self._replay_live_pose()
        self.last_error = ""
        # Dispatch always publishes the ack. Skip a second storm on no-op.
        return self.status(publish=changed)

    def _replay_live_pose(self) -> None:
        snap = self._live_pose
        if snap is None or self._finishing:
            return
        kind, frame, mixed, _stored_rest = snap
        rest = self.rest_pts
        if mixed is None or rest is None or not hasattr(frame, "head"):
            return
        head = frame.head
        pose = frame.pose
        if kind == "osf":
            posed = self._rig.apply(mixed, rest, head, pose)
            posed = self._expr.place_brows(posed, rest, self._rig)
        else:
            posed = self._rig.apply(mixed, rest, head, pose)
        self._finish_live(frame, posed)

    def set_mouth_point(self, body: object) -> dict[str, object]:
        if not isinstance(body, dict):
            return self.status(publish=True)
        try:
            idx = int(body.get("id", -1))
        except (TypeError, ValueError):
            return self.status(publish=True)
        if "on" in body:
            mouth_bits.set_on(idx, bool(body.get("on")))
        if "to" in body:
            raw = body.get("to")
            if raw is None or raw == "":
                mouth_bits.set_to(idx, None)
            else:
                try:
                    mouth_bits.set_to(idx, int(raw))
                except (TypeError, ValueError):
                    pass
        self.last_error = ""
        return self.status(publish=True)

    def set_eye_point(self, body: object) -> dict[str, object]:
        if not isinstance(body, dict):
            return self.status(publish=True)
        try:
            idx = int(body.get("id", -1))
        except (TypeError, ValueError):
            return self.status(publish=True)
        if "on" in body:
            eye_bits.set_on(idx, bool(body.get("on")))
        if "to" in body:
            raw = body.get("to")
            if raw is None or raw == "":
                eye_bits.set_to(idx, None)
            else:
                try:
                    eye_bits.set_to(idx, int(raw))
                except (TypeError, ValueError):
                    pass
        self.last_error = ""
        return self.status(publish=True)

    def set_skeleton_point(self, body: object) -> dict[str, object]:
        if self._osf.running:
            self.last_error = "Stop OSF before editing the skeleton"
            return self.status(publish=True)
        if not isinstance(body, dict):
            return self.status(publish=True)
        try:
            idx = int(body.get("id", -1))
            x = float(body.get("x"))
            y = float(body.get("y"))
        except (TypeError, ValueError):
            self.last_error = "Invalid skeleton point"
            return self.status(publish=True)
        if idx not in {31, 32, 33, 34, 35, 36}:
            self.last_error = "Only skeleton points 31–36 are editable"
            return self.status(publish=True)
        changed = False
        for joint in self._skeleton_rest:
            if int(joint.get("id", -1)) == idx:
                joint["x"] = round(x, 1)
                joint["y"] = round(y, 1)
                changed = True
                break
        if changed:
            self._skeleton = [dict(j) for j in self._skeleton_rest]
            self._save_parts()
            self.last_error = ""
        return self.status(publish=True)

    def set_rest_point(self, body: object) -> dict[str, object]:
        """Move one rest face point (0–27) or iris (28–29) on the still.

        The rest itself moves, so any overlay nudge on the same point is
        dropped — otherwise the edit would be applied twice.
        """
        if not isinstance(body, dict):
            return self.status(publish=True)
        try:
            idx = int(body.get("id", -1))
            x = float(body.get("x"))
            y = float(body.get("y"))
        except (TypeError, ValueError):
            self.last_error = "Invalid rest point"
            return self.status(publish=True)
        if not 0 <= idx <= 29:
            self.last_error = "Only face points 0–27 and irises 28–29 are rest points"
            return self.status(publish=True)
        if idx < 28:
            if self.rest_pts is None or idx >= len(self.rest_pts):
                self.last_error = "Track the still before moving its points"
                return self.status(publish=True)
            pts = self.rest_pts.copy()
            pts[idx, 0] = x
            pts[idx, 1] = y
            if pts.shape[1] > 2:
                pts[idx, 2] = max(float(pts[idx, 2]), 0.85)
            book.rebase(pts)
            rest_hair = rig_rest_hair(self._hair_rig) or list(self._hair)
            with self._lock:
                self.rest_pts = pts
                self._hair_rig = build_hair_rig(rest_hair, pts)
                self._point_offsets = clear_offsets(self._point_offsets, idx)
            if self.source_bgr is not None:
                self.overlay_bgr = draw_label28(self.source_bgr, pts)
        else:
            rows = [dict(row) for row in self._iris_rest if int(row.get("id", -1)) != idx]
            rows.append({"id": idx, "x": round(x, 3), "y": round(y, 3), "score": 1.0, "visible": True})
            rows.sort(key=lambda row: int(row.get("id", -1)))
            method = self._iris_rest_method if self._iris_rest_method not in ("", "none") else "manual"
            self._set_iris_rest(rows, method)
            with self._lock:
                self._point_offsets = clear_offsets(self._point_offsets, idx)
        self._save_parts()
        self.last_error = ""
        return self.status(publish=True)

    def set_hair(self, body: object) -> dict[str, object]:
        """Replace rest hair polygons. Points are character pixels."""
        if not isinstance(body, dict):
            return self.status(publish=True)
        raw = body.get("hair")
        if not isinstance(raw, list):
            self.last_error = "Hair list missing"
            return self.status(publish=True)
        cleaned: list[dict[str, object]] = []
        for seg in raw:
            if not isinstance(seg, dict):
                continue
            cls = str(seg.get("class") or "")
            poly = seg.get("polygon") or []
            if cls not in HAIR_CLASSES or not isinstance(poly, list):
                continue
            pts: list[list[float]] = []
            for vertex in poly:
                if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                    continue
                try:
                    pts.append([round(float(vertex[0]), 1), round(float(vertex[1]), 1)])
                except (TypeError, ValueError):
                    continue
            if len(pts) >= 3:
                cleaned.append({"class": cls, "polygon": pts, "score": 1.0})
        rig = build_hair_rig(cleaned, self.rest_pts) if self.rest_pts is not None else None
        with self._lock:
            self._hair = cleaned
            self._hair_rig = rig
            self._smooth_hair = []
        self._save_parts()
        self.last_error = ""
        return self.status(publish=True)

    def set_point(self, body: object) -> dict[str, object]:
        if not isinstance(body, dict):
            return self.status(publish=True)
        try:
            idx = int(body.get("id", -1))
            x = float(body.get("x"))
            y = float(body.get("y"))
        except (TypeError, ValueError):
            self.last_error = "Invalid overlay point"
            return self.status(publish=True)
        with self._lock:
            cur = self._unoffset_xy(idx)
            if cur is None:
                self.last_error = "No overlay point to nudge"
                return self.status(publish=True)
            self._point_offsets = nudge_offset(self._point_offsets, idx, cur, (x, y))
            self.last_error = ""
        self._save_parts()
        return self.status(publish=True)

    def reset_points(self, body: object) -> dict[str, object]:
        idx = None
        if isinstance(body, dict) and body.get("id") not in (None, ""):
            try:
                idx = int(body.get("id"))
            except (TypeError, ValueError):
                self.last_error = "Invalid overlay point"
                return self.status(publish=True)
        with self._lock:
            self._point_offsets = clear_offsets(self._point_offsets, idx)
            self.last_error = ""
        self._save_parts()
        return self.status(publish=True)

    def start_calibrate(self, name: str) -> dict[str, object]:
        if not (self._osf.running or self._ifm.running):
            self.last_error = "Start tracking first, then hold the face and calibrate"
            return self.status(publish=True)
        try:
            calibrator.start(str(name))
        except ValueError as exc:
            self.last_error = str(exc)
            return self.status(publish=True)
        if str(name) == "rest":
            if self._ifm.running:
                frame = self._canonical_ifm(self._ifm.latest)
            else:
                frame = self._osf.latest
            self._capture_rest_look(frame, force=True)
            self._save_parts()
        self.last_error = ""
        return self.status(publish=True)

    def reset_calibrate(self) -> dict[str, object]:
        calibrator.clear()
        self.last_error = ""
        return self.status(publish=True)

    def record_movement(self, on: bool) -> dict[str, object]:
        """Start or stop a benchmark take of the live character overlay."""
        live = self._osf.running or self._ifm.running
        if on:
            if not live:
                self.last_error = "Start tracking before recording movement"
                return self.status(publish=True)
            if self.source_bgr is None:
                self.last_error = "Load a reference image before recording movement"
                return self.status(publish=True)
            try:
                self._recorder.start(self.source_bgr, str(source_path()))
            except ValueError as exc:
                self.last_error = str(exc)
                return self.status(publish=True)
            self.last_error = ""
            return self.status(publish=True)
        saved = self._recorder.stop()
        if saved.get("record_error"):
            self.last_error = str(saved["record_error"])
        else:
            self.last_error = ""
        return self.status(publish=True)

    def stop_live(self) -> dict[str, object]:
        if self._recorder.recording:
            self._recorder.stop()
        self._osf.stop()
        self._ifm.stop()
        self._save_ifm()
        self._rig.reset()
        self._expr.reset()
        with self._lock:
            self._live_pts = None
            self._smooth_pts = None
            self._smooth_hair = []
            self._smooth_skeleton = []
            self._iris_cam = []
            self._look = None
            self._restore_iris_rest()
            self._mouth_box = None
            self._mouth_cage = None
            self._weights = empty_weights()
            self.camera_bgr = None
        pts = book.current()
        if pts is None:
            pts = self.rest_pts
        if pts is not None:
            self._paint(pts)
            if self._hair_rig is not None:
                self._hair = follow_hair(self._hair_rig, pts)
            if self._skeleton_rest:
                self._skeleton = [dict(j) for j in self._skeleton_rest]
        if self.last_tracker in ("osf", "ifm"):
            self.last_tracker = "anime"
        return self.status(publish=True)

    def _on_osf(self, frame: OsfFrame) -> None:
        ifm = getattr(frame, "source", "") == "ifm"
        if frame.pts_3d is None and not (ifm and int(frame.faces or 0) > 0):
            with self._lock:
                self.camera_bgr = frame.camera_jpeg or None
                self.last_faces = frame.faces
                self.last_ms = frame.ms
                self._iris_cam = []
                self._look = None
                self._restore_iris_rest()
                if frame.error:
                    self.last_error = frame.error
            self._publish(frame=True)
            return
        if ifm:
            frame = self._canonical_ifm(frame)
            posed = self._pose_ifm(frame)
        else:
            posed = self._pose_osf(frame)
        self._finish_live(frame, posed)

    @staticmethod
    def _canonical_ifm(frame: OsfFrame) -> OsfFrame:
        """ARKit sides are the person's own. Write them into the canonical
        camera frame (person-right = image-left) so the one selfie rule
        applies the same as OSF. Returns a copy; ``latest`` stays raw."""
        look = ifm_look_canonical(getattr(frame, "look", None))
        return replace(
            frame,
            blink=ifm_canonical(frame.blink),
            brow=ifm_canonical(getattr(frame, "brow", None)),
            look=look if look is not None else frame.look,
        )

    def _pose_ifm(self, frame: OsfFrame) -> np.ndarray | None:
        mixed = None
        if feel.use_visemes():
            mixed = book.mix(frame.weights)
        selfie = self.selfie
        driven = drive_ifm(
            self.rest_pts,
            frame.weights,
            to_screen(frame.blink, selfie),
            to_screen(getattr(frame, "brow", None), selfie),
            mixed=mixed,
        )
        self._live_pose = ("ifm", frame, None if driven is None else driven.copy(), self.rest_pts)
        posed = self._rig.apply(driven, self.rest_pts, frame.head, frame.pose)
        return posed

    def _pose_osf(self, frame: OsfFrame) -> np.ndarray | None:
        use_visemes = feel.use_visemes()
        viseme_pts = book.mix(frame.weights if use_visemes else None)
        if viseme_pts is None and self.rest_pts is not None:
            viseme_pts = self.rest_pts.copy()
        mixed = self._expr.apply(
            viseme_pts,
            self.rest_pts,
            frame.pts_3d,
            to_screen(frame.blink, self.selfie),
            mouth_pts=frame.mouth_2d,
            keep_mouth=use_visemes,
        )
        self._live_pose = ("osf", frame, None if mixed is None else mixed.copy(), self.rest_pts)
        posed = self._rig.apply(mixed, self.rest_pts, frame.head, frame.pose)
        posed = self._expr.place_brows(posed, self.rest_pts, self._rig)
        return posed

    def _finish_live(self, frame: OsfFrame, posed: np.ndarray | None) -> None:
        if self._finishing:
            self._snap_smooth = True
            return
        self._finishing = True
        try:
            self._finish_live_body(frame, posed)
        finally:
            self._finishing = False

    def _finish_live_body(self, frame: OsfFrame, posed: np.ndarray | None) -> None:
        snap = self._snap_smooth
        self._snap_smooth = False
        alpha = 1.0 if snap else feel.alpha()
        if posed is not None:
            if self._smooth_pts is None or self._smooth_pts.shape != posed.shape:
                self._smooth_pts = posed.copy()
            else:
                self._smooth_pts += alpha * (posed - self._smooth_pts)
            posed = self._smooth_pts.copy()
        look = getattr(frame, "look", None)
        if isinstance(look, dict) or (frame.lms_xy is not None and frame.iris_cam):
            with self._lock:
                missing_look = "x" not in self._look_rest and "y" not in self._look_rest
                missing_cam = (
                    frame.lms_xy is not None
                    and frame.iris_cam
                    and not self._has_cam_rest()
                )
            # Do not lock a look-up as rest. Set Rest still force-captures.
            if missing_look and look_quiet(look if isinstance(look, dict) else None):
                self._capture_rest_look(frame, force=False)
            elif missing_cam:
                self._capture_rest_look(frame, force=False)
        rest_iris = [dict(row) for row in self._iris_rest]
        rest_look = dict(self._look_rest)
        rest_pts = None if self.rest_pts is None else self.rest_pts.copy()
        selfie = self.selfie
        blink_screen = to_screen(frame.blink, selfie)
        iris_rows, iris_method = retarget_iris(
            posed,
            cam_lms=frame.lms_xy if getattr(frame, "source", "") != "ifm" else None,
            cam_iris=frame.iris_cam if getattr(frame, "source", "") != "ifm" else None,
            look=frame.look,
            blink=blink_screen,
            rest_iris=rest_iris,
            rest_look=rest_look,
            rest_pts=rest_pts,
            gaze_gain=feel.gaze_gain(),
            selfie=selfie,
            max_look_x=feel.max_look_x(),
            max_look_y=feel.max_look_y(),
        )
        with self._lock:
            self._live_pts = posed
            if getattr(frame, "source", "") == "ifm":
                self._mouth_box = None
                self._mouth_cage = None
            else:
                self._mouth_box = self._expr.mouth_box()
                self._mouth_cage = self._expr.mouth_cage()
            if posed is not None:
                followed = follow_hair(self._hair_rig, posed, self._rig)
                if followed:
                    self._hair = self._smooth_records(
                        self._smooth_hair, followed, alpha, polygons=True
                    )
                    self._smooth_hair = [dict(part) for part in self._hair]
                if self._skeleton_rest:
                    skeleton = follow_skeleton(
                        self._skeleton_rest,
                        posed,
                        head={
                            "yaw": float(np.degrees(self._rig._yaw_r)),
                            "pitch": float(np.degrees(self._rig._pitch_r)),
                            "roll": float(np.degrees(self._rig._roll_r)),
                        },
                        place=self._rig.place(),
                        rig=self._rig,
                    )
                    self._skeleton = self._smooth_records(
                        self._smooth_skeleton, skeleton, alpha
                    )
                    self._smooth_skeleton = [dict(joint) for joint in self._skeleton]
            if iris_rows:
                self._iris = self._smooth_records(
                    self._smooth_iris, iris_rows, feel.gaze_alpha()
                )
                self._smooth_iris = [dict(row) for row in self._iris]
                self._iris_method = iris_method
            else:
                self._iris = []
                self._smooth_iris = []
                self._iris_method = "none"
            # Clamp after pose / skeleton / hair / iris are built, before publish.
            posed, self._skeleton, self._iris, self._hair = self._apply_travel_limits(
                posed,
                list(self._skeleton),
                list(self._iris),
                list(self._hair),
            )
            self._live_pts = posed
            debug = raw_debug(frame.iris_cam, look)
            self._iris_cam = list(debug["iris_cam"])
            packed_look = debug["look"]
            self._look = dict(packed_look) if isinstance(packed_look, dict) else None
            self._weights = dict(frame.weights)
            self._head = dict(frame.head)
            # Meters read by character screen side, same as the overlay.
            self._blink = dict(blink_screen)
            self.camera_bgr = frame.camera_jpeg or None
            self.last_faces = frame.faces
            self.last_ms = frame.ms
            if frame.error:
                self.last_error = frame.error
            elif not self._recorder.recording:
                self.last_error = str(calibrator.payload().get("error") or "")
            # #region agent log
            _now = time.perf_counter()
            if debug_log.ENABLED and _now - float(getattr(self, "_dbg_t", 0.0)) >= 0.5:
                self._dbg_t = _now
                _prev = getattr(self, "_dbg_pts", None)
                _motion = 0.0
                if posed is not None and _prev is not None and getattr(_prev, "shape", None) == posed.shape:
                    _motion = float(np.max(np.abs(posed[:, :2] - _prev[:, :2])))
                if posed is not None:
                    self._dbg_pts = posed.copy()
                _head = frame.head if isinstance(frame.head, dict) else {}
                _pose = frame.pose if isinstance(frame.pose, dict) else {}
                debug_log.log(
                    "E",
                    "face.py:_finish_live_body",
                    "overlay",
                    {
                        "faces": int(frame.faces or 0),
                        "ms": round(float(frame.ms or 0.0), 1),
                        "yaw": _head.get("yaw"),
                        "pitch": _head.get("pitch"),
                        "pose_ok": _pose.get("ok"),
                        "cx": _pose.get("cx"),
                        "cy": _pose.get("cy"),
                        "motion": round(_motion, 2),
                        "error": str(frame.error or ""),
                    },
                )
            # #endregion
            if self._recorder.recording and posed is not None and self.source_bgr is not None:
                offsets = dict(self._point_offsets)
                h, w = int(self.source_bgr.shape[0]), int(self.source_bgr.shape[1])
                points = offset_points(pts_to_json(posed), offsets)
                self._recorder.note(
                    points,
                    offset_rows(list(self._skeleton), offsets),
                    offset_rows(list(self._iris), offsets),
                    list(self._hair),
                    dict(self._weights),
                    w,
                    h,
                )
        self._publish(frame=True)

    @staticmethod
    def _smooth_records(
        previous: list[dict[str, object]],
        current: list[dict[str, object]],
        alpha: float,
        *,
        polygons: bool = False,
    ) -> list[dict[str, object]]:
        """Apply the global Smooth value to every rendered overlay point."""
        if not previous or len(previous) != len(current):
            return [dict(item) for item in current]
        out: list[dict[str, object]] = []
        for old, new in zip(previous, current):
            record = dict(new)
            if polygons:
                old_poly = old.get("polygon")
                new_poly = new.get("polygon")
                if (
                    isinstance(old_poly, list)
                    and isinstance(new_poly, list)
                    and len(old_poly) == len(new_poly)
                    and old.get("class") == new.get("class")
                ):
                    record["polygon"] = [
                        [
                            float(a[0]) + alpha * (float(b[0]) - float(a[0])),
                            float(a[1]) + alpha * (float(b[1]) - float(a[1])),
                        ]
                        for a, b in zip(old_poly, new_poly)
                    ]
            else:
                record["x"] = float(old["x"]) + alpha * (
                    float(new["x"]) - float(old["x"])
                )
                record["y"] = float(old["y"]) + alpha * (
                    float(new["y"]) - float(old["y"])
                )
            out.append(record)
        return out

    def _unoffset_xy(self, idx: int) -> tuple[float, float] | None:
        if 0 <= idx < 28:
            live = self._osf.running or self._ifm.running
            pts = self._live_pts if live and self._live_pts is not None else None
            if pts is None:
                pts = book.current()
            if pts is None:
                pts = self.rest_pts
            if pts is None or idx >= len(pts):
                return None
            return float(pts[idx, 0]), float(pts[idx, 1])
        if idx in (28, 29):
            rows = self._iris or self._iris_rest
            for row in rows:
                try:
                    if int(row.get("id", -1)) != idx:
                        continue
                    return float(row.get("x") or 0.0), float(row.get("y") or 0.0)
                except (TypeError, ValueError, AttributeError):
                    continue
            return None
        for row in self._skeleton:
            try:
                if int(row.get("id", -1)) != idx:
                    continue
                return float(row.get("x") or 0.0), float(row.get("y") or 0.0)
            except (TypeError, ValueError, AttributeError):
                continue
        return None

    def _has_cam_rest(self) -> bool:
        r = self._look_rest.get("r")
        l = self._look_rest.get("l")
        return isinstance(r, dict) or isinstance(l, dict)

    def _capture_rest_look(self, frame: OsfFrame, *, force: bool = False) -> None:
        changed = False
        look = getattr(frame, "look", None)
        with self._lock:
            if isinstance(look, dict) and (force or ("x" not in self._look_rest and "y" not in self._look_rest)):
                try:
                    self._look_rest["x"] = float(look.get("x") or 0.0)
                    self._look_rest["y"] = float(look.get("y") or 0.0)
                    changed = True
                except (TypeError, ValueError):
                    pass
            lms = getattr(frame, "lms_xy", None)
            iris = getattr(frame, "iris_cam", None)
            if lms is not None and iris and (force or not self._has_cam_rest()):
                right, left = payload_to_hits(iris)
                snap = rest_look_from_cam(lms, right, left)
                if snap:
                    self._look_rest.update(snap)
                    changed = True
        if changed:
            self._save_parts()

    def _set_iris_rest(self, rows: list[dict[str, object]], method: str) -> None:
        packed = [dict(row) for row in rows]
        with self._lock:
            self._iris_rest = packed
            self._iris_rest_method = str(method)
            self._iris = [dict(row) for row in packed]
            self._smooth_iris = [dict(row) for row in packed]
            self._iris_method = str(method)

    def _restore_iris_rest(self) -> None:
        self._iris = [dict(row) for row in self._iris_rest]
        self._smooth_iris = [dict(row) for row in self._iris_rest]
        self._iris_method = str(self._iris_rest_method)

    def _apply_still_iris(self, frame: np.ndarray, pts: np.ndarray) -> None:
        try:
            rows, method = track_still(frame, pts)
        except Exception as exc:
            print(f"[track-lab] iris still failed: {exc}", flush=True)
            rows, method = [], "none"
        self._set_iris_rest(rows, method)

    def _capture_parts(self, frame: np.ndarray, pts: np.ndarray) -> None:
        """Hair model, then skeleton. Hair polygons are not cut by the face mesh."""
        try:
            segs = detect_hair(frame)
        except Exception as exc:
            segs = []
            if not self.last_error:
                self.last_error = f"Hair: {exc}"
        self._hair = segs
        self._hair_rig = build_hair_rig(segs, pts)
        locked = skeleton_from_still(pts, frame)
        self._skeleton = locked
        self._skeleton_rest = [dict(j) for j in locked]
        self._save_parts()

    def _save_parts(self) -> None:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            PARTS_PATH.write_text(
                json.dumps(
                    {
                        # _hair is the followed pose while live. Save rest.
                        "hair": rig_rest_hair(self._hair_rig) or list(self._hair),
                        "skeleton": list(self._skeleton_rest),
                        "iris": list(self._iris_rest),
                        "iris_method": self._iris_rest_method,
                        "look_rest": dict(self._look_rest),
                        "point_offsets": dump_offsets(self._point_offsets),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load_parts(self) -> None:
        if not PARTS_PATH.is_file():
            return
        try:
            data = json.loads(PARTS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        hair = data.get("hair")
        body = data.get("skeleton")
        if isinstance(hair, list) and hair and not self._hair:
            self._hair = hair
            if self.rest_pts is not None:
                self._hair_rig = build_hair_rig(hair, self.rest_pts)
        if isinstance(body, list) and body and not self._skeleton_rest:
            manual = [
                dict(j)
                for j in body
                if isinstance(j, dict) and 31 <= int(j.get("id", -1)) <= 36
            ]
            self._skeleton = manual
            self._skeleton_rest = [dict(j) for j in manual]
        iris = data.get("iris")
        if isinstance(iris, list) and iris and not self._iris_rest:
            rows = []
            for row in iris:
                if not isinstance(row, dict):
                    continue
                try:
                    idx = int(row.get("id", -1))
                except (TypeError, ValueError):
                    continue
                if idx not in (28, 29):
                    continue
                rows.append(dict(row))
            if rows:
                method = str(data.get("iris_method") or "iris_pose")
                self._set_iris_rest(rows, method)
        look_rest = data.get("look_rest")
        if isinstance(look_rest, dict) and not self._look_rest:
            packed: dict[str, object] = {}
            for key in ("r", "l"):
                side = look_rest.get(key)
                if not isinstance(side, dict):
                    continue
                try:
                    packed[key] = {
                        "nx": float(side.get("nx") or 0.0),
                        "ny": float(side.get("ny") or 0.0),
                    }
                except (TypeError, ValueError):
                    continue
            try:
                if "x" in look_rest:
                    packed["x"] = float(look_rest.get("x") or 0.0)
                if "y" in look_rest:
                    packed["y"] = float(look_rest.get("y") or 0.0)
            except (TypeError, ValueError):
                pass
            if packed:
                self._look_rest = packed
        if not self._point_offsets:
            self._point_offsets = parse_offsets(data.get("point_offsets"))

    def _load_ifm(self) -> None:
        if not IFM_PATH.is_file():
            return
        try:
            data = json.loads(IFM_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        source = data.get("source")
        if source in ("camera", "ifm"):
            self._source = source
        host = data.get("host")
        port = data.get("port")
        try:
            self._ifm.configure(
                str(host) if host is not None else None,
                int(port) if port is not None else None,
            )
        except (TypeError, ValueError):
            pass
        peer = data.get("last_peer")
        if isinstance(peer, str) and peer.strip():
            self._ifm.last_peer = peer.strip()
        if "mirror" in data:
            self._apply_mirror(bool(data.get("mirror")))

    def _save_ifm(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        IFM_PATH.write_text(
            json.dumps(
                {
                    "source": self._source,
                    "host": self._ifm.host,
                    "last_peer": self._ifm.last_peer,
                    "port": int(self._ifm.port or DEFAULT_PORT),
                    "mirror": bool(self._mirror),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _paint(self, pts: np.ndarray) -> None:
        if self.source_bgr is None:
            return
        self.overlay_bgr = draw_label28(self.source_bgr, pts)

    def _authored_status(self, pts: np.ndarray) -> dict[str, object]:
        """Still-editor snapshot: the saved shape, not shape plus leftover nudges."""
        payload = self.status(publish=False)
        payload["points"] = pts_to_json(pts)
        self._publish(payload)
        return payload

    def _ensure_source(self, *, force: bool = False) -> None:
        path = source_path()
        if not path.is_file():
            self.source_bgr = None
            return
        mtime = path.stat().st_mtime
        if not force and self.source_bgr is not None and mtime == self.source_mtime:
            return
        image = _read_bgr(path)
        if image is None:
            self.source_bgr = None
            self.last_error = f"Could not read {path.name}"
            return
        self.source_bgr = image
        self.source_mtime = mtime


bench = FaceBench()
