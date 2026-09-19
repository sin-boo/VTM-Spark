"""Camera discovery + capture for Live Poser.

On Windows, uses OpenSeeFace DirectShow (dshowcapture) so virtual cameras
(OBS Virtual Camera, WarudoCam, nizima LIVE, DroidCam, etc.) appear by name.
Falls back to OpenCV CAP_DSHOW when needed.

IMPORTANT: DirectShow/COM must be used on a thread where CoInitialize(Ex) has
run. The UI worker thread calls ensure_com() before listing/opening.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT.parent
DEFAULT_OSF = TOOLS / 'openseeface'

_com_local = threading.local()


def resolve_openseeface(path: Path | None = None) -> Path:
    if path is not None:
        osf = Path(path).resolve()
        if not (osf / 'tracker.py').is_file():
            raise FileNotFoundError(
                f'OpenSeeFace not found at {osf}\n'
                'Install OpenSeeFace under vendor/tools/openseeface or pass --osf-dir.'
            )
        return osf
    for cand in (
        TOOLS / 'openseeface',
    ):
        if (cand / 'tracker.py').is_file():
            return cand.resolve()
    raise FileNotFoundError(
        'OpenSeeFace not found.\n'
        'Expected vendor/tools/openseeface.'
    )


def ensure_osf_on_path(osf: Path) -> None:
    s = str(osf)
    if s not in sys.path:
        sys.path.insert(0, s)


def ensure_com() -> None:
    """Initialize COM on the current thread (required for DirectShow)."""
    if os.name != 'nt':
        return
    if getattr(_com_local, 'ready', False):
        return
    import ctypes

    # COINIT_MULTITHREADED = 0x0 — safe for background capture threads
    hr = ctypes.windll.ole32.CoInitializeEx(None, 0x0)
    # S_OK (0), S_FALSE (1) = already inited; RPC_E_CHANGED_MODE (-2147417850) = different mode
    _com_local.ready = True
    _com_local.hr = hr


def uninit_com() -> None:
    if os.name != 'nt':
        return
    if not getattr(_com_local, 'ready', False):
        return
    hr = getattr(_com_local, 'hr', 1)
    # Only uninit if we successfully initialized on this thread (S_OK)
    if hr == 0:
        import ctypes

        ctypes.windll.ole32.CoUninitialize()
    _com_local.ready = False


@dataclass
class CameraInfo:
    index: int
    name: str
    backend: str  # 'dshow' | 'opencv'
    raw: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        tag = ''
        lower = self.name.lower()
        if any(
            k in lower
            for k in ('virtual', 'obs', 'warudo', 'nizima', 'unity', 'manycam', 'snap', 'droidcam')
        ):
            tag = ' [virtual]'
        return f'{self.index}: {self.name}{tag}'


def list_cameras(osf: Path | None = None) -> list[CameraInfo]:
    """List capture devices, including Windows virtual cameras via DirectShow."""
    ensure_com()
    if os.name == 'nt':
        try:
            cams = _list_dshow_cameras(osf)
            if cams:
                return cams
        except Exception as exc:
            print(f'DirectShow camera list failed ({exc}); falling back to OpenCV')
    return _list_opencv_cameras()


def _list_dshow_cameras(osf: Path | None = None) -> list[CameraInfo]:
    osf_path = resolve_openseeface(osf)
    ensure_osf_on_path(osf_path)
    import dshowcapture  # noqa: E402

    ensure_com()
    cap = dshowcapture.DShowCapture()
    try:
        # Enumerate first — get_info alone can return [] on some threads/states.
        try:
            cap.get_devices()
        except Exception:
            pass
        info = cap.get_info() or []
        cameras: list[CameraInfo] = []
        for i, cam in enumerate(info):
            idx = int(cam.get('index', cam.get('id', i)))
            name = str(cam.get('name') or f'Camera {idx}')
            if not _is_camera_name(name):
                continue
            cameras.append(
                CameraInfo(
                    index=idx,
                    name=name,
                    backend='dshow',
                    raw=cam,
                )
            )
        return cameras
    finally:
        try:
            cap.destroy_capture()
        except Exception:
            pass


_AUDIO_DEVICE = ('microphone', 'stereo mix', 'wave in', 'what u hear', 'speaks')


def _is_camera_name(name: str) -> bool:
    n = name.lower()
    if any(key in n for key in ('camera', 'webcam', 'droidcam', 'obs', 'virtual', 'capture')):
        return True
    return not any(key in n for key in _AUDIO_DEVICE)


def _no_mic_params() -> list[int]:
    """Do not let OpenCV take the Windows microphone with the webcam."""
    params: list[int] = []
    video = getattr(cv2, 'CAP_PROP_VIDEO_STREAM', None)
    if video is not None:
        params.extend([int(video), 0])
    stream = getattr(cv2, 'CAP_PROP_AUDIO_STREAM', None)
    if stream is not None:
        params.extend([int(stream), -1])
    return params


def _open_cv_index(index: int, backend: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture()
    params = _no_mic_params()
    if params:
        cap.open(int(index), int(backend), params)
    else:
        cap.open(int(index), int(backend))
    return cap


def _list_opencv_cameras(max_probe: int = 12) -> list[CameraInfo]:
    ensure_com()
    cameras: list[CameraInfo] = []
    for i in range(max_probe):
        cap = None
        try:
            if os.name == 'nt':
                cap = _open_cv_index(i, cv2.CAP_DSHOW)
            else:
                cap = _open_cv_index(i, cv2.CAP_ANY)
            if cap is None or not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                cameras.append(CameraInfo(index=i, name=f'Camera {i}', backend='opencv'))
        except Exception:
            continue
        finally:
            if cap is not None:
                cap.release()
    return cameras


def _frame_ok(frame: np.ndarray | None, min_mean: float = 5.0, min_std: float = 2.0) -> bool:
    """Reject black / solid frames (common with DroidCam default DirectShow mode)."""
    if frame is None or getattr(frame, 'size', 0) == 0:
        return False
    h, w = frame.shape[:2]
    step = max(1, min(h, w) // 64)
    sample = frame[::step, ::step]
    mean = float(np.mean(sample))
    std = float(np.std(sample))
    return mean >= min_mean and std >= min_std


def _dshow_capability_attempts(
    camera_index: int,
    width: int,
    height: int,
    preferred_dcap: int | None,
    *,
    max_unique: int = 6,
    prefer_mid_res: bool = False,
) -> list[int | None]:
    """Build an ordered list of DirectShow capability IDs to try.

    DroidCam's default / first 1280x720 mode often returns pure-black frames via
    libdshowcapture, while other capability lines (e.g. 640x480 or alt 1280x720)
    work fine. So we never stop at the first open — caller validates pixels.

    DroidCam often exposes the same WxH many times (fps / colorspace variants).
    Opening every line wedges the device, so we keep one id per resolution.
    """
    import dshowcapture  # noqa: E402

    attempts: list[int | None] = []
    if preferred_dcap is not None:
        attempts.append(preferred_dcap)

    info = []
    probe = dshowcapture.DShowCapture()
    try:
        try:
            probe.get_devices()
        except Exception:
            pass
        info = probe.get_info() or []
    finally:
        try:
            probe.destroy_capture()
        except Exception:
            pass

    caps = []
    if 0 <= camera_index < len(info):
        caps = list(info[camera_index].get('caps') or [])

    def score(cap: dict) -> tuple:
        cx = int(cap.get('minCX') or 0)
        cy = int(cap.get('minCY') or 0)
        # DirectShow intervals are 100ns units; smaller minInterval => higher fps.
        min_iv = int(cap.get('minInterval') or 333333)
        if prefer_mid_res:
            # DroidCam: try 640/960 before 1280+ (default HD is often pure black).
            mid = abs(cx - 640) + abs(cy - 480)
            size_dist = abs(cx - width) + abs(cy - height)
            huge = 1 if (cx * cy) >= (1280 * 720) else 0
            return (huge, mid, size_dist, cx * cy, min_iv)
        # Prefer requested size, then mid-res (640/960/1280), avoid huge first.
        size_dist = abs(cx - width) + abs(cy - height)
        mid = abs(cx - 960) + abs(cy - 720)
        return (size_dist, mid, cx * cy, min_iv)

    # One capability id per (width, height) — DroidCam lists 4+ variants each.
    best_by_size: dict[tuple[int, int], dict] = {}
    for cap in caps:
        cx = int(cap.get('minCX') or 0)
        cy = int(cap.get('minCY') or 0)
        if cx <= 0 or cy <= 0:
            continue
        key = (cx, cy)
        prev = best_by_size.get(key)
        if prev is None or score(cap) < score(prev):
            best_by_size[key] = cap

    ranked = sorted(best_by_size.values(), key=score)
    for cap in ranked[:max_unique]:
        cid = cap.get('id')
        if cid is None:
            continue
        cid = int(cid)
        if cid not in attempts:
            attempts.append(cid)

    # Explicit None = capture_device(width,height,fps) negotiation
    if None not in attempts:
        attempts.append(None)
    return attempts


class CameraCapture:
    """Unified frame source (DirectShow via OpenSeeFace, or OpenCV)."""

    def __init__(
        self,
        camera_index: int,
        width: int = 640,
        height: int = 360,
        fps: int = 30,
        backend: str = 'auto',
        osf: Path | None = None,
        dcap: int | None = None,
        camera_name: str | None = None,
    ):
        ensure_com()
        self.name = camera_name or f'Camera {camera_index}'
        self.width = width
        self.height = height
        self.fps = fps
        self._reader = None
        self._cap = None
        self.backend = backend
        self.camera_index = camera_index
        self.dcap_used: int | None = None
        self.last_health = 'starting'
        self.last_error: str | None = None

        errors: list[str] = []

        # Prefer DirectShow native reader for named virtual cams on Windows.
        if backend in ('auto', 'dshow') and os.name == 'nt':
            try:
                self._open_dshow(camera_index, width, height, fps, osf, dcap)
                self.backend = 'dshow'
                return
            except Exception as exc:
                errors.append(f'dshow: {exc}')

        try:
            self._open_opencv(camera_index, width, height, fps)
            self.backend = 'opencv'
            return
        except Exception as exc:
            errors.append(f'opencv: {exc}')

        raise RuntimeError(
            f'Could not open camera {camera_index}'
            + (f' ({camera_name})' if camera_name else '')
            + ': '
            + ' | '.join(errors)
        )

    def _accept_reader(self, reader, dcap_label, *, warm_reads: int = 40) -> bool:
        """Warm up and require non-black frames before accepting a capture mode."""
        import time

        if not reader.is_open():
            reader.close()
            return False

        # DroidCam often needs >0.5s after a mode switch before real pixels arrive.
        best = None
        for i in range(warm_reads):
            ok, frame = reader.read()
            if ok and frame is not None and _frame_ok(frame):
                best = frame
                break
            time.sleep(0.05 if i < 10 else 0.08)

        if best is None:
            reader.close()
            return False

        self._reader = reader
        self.name = reader.name
        self.width = reader.width
        self.height = reader.height
        self.fps = reader.fps
        self.dcap_used = dcap_label if isinstance(dcap_label, int) else None
        return True

    def _open_dshow(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        osf: Path | None,
        dcap: int | None,
    ) -> None:
        import time

        osf_path = resolve_openseeface(osf)
        ensure_osf_on_path(osf_path)
        ensure_com()
        from input_reader import DShowCaptureReader  # noqa: E402

        attempts = _dshow_capability_attempts(
            camera_index,
            width,
            height,
            dcap,
            max_unique=4 if 'droidcam' in (self.name or '').lower() else 6,
            prefer_mid_res='droidcam' in (self.name or '').lower(),
        )
        # Never lead with dcap=-1 for cams like DroidCam — default mode can be black.
        # Keep it as a late fallback only.
        attempts = [a for a in attempts if a != -1] + [-1]

        last_err: Exception | None = None
        tried: list[str] = []
        black_streak = 0
        for i, attempt_dcap in enumerate(attempts):
            try:
                reader = DShowCaptureReader(
                    int(camera_index),
                    width,
                    height,
                    fps,
                    use_dshowcapture=True,
                    dcap=attempt_dcap,
                )
                if self._accept_reader(reader, attempt_dcap):
                    print(
                        f'Camera OK: "{self.name}" dcap={attempt_dcap} '
                        f'{self.width}x{self.height} (non-black frame verified)'
                    )
                    return
                tried.append(f'{attempt_dcap}:black')
                black_streak += 1
            except Exception as exc:
                last_err = exc
                tried.append(f'{attempt_dcap}:err')
                black_streak = 0
                continue
            # Let DroidCam / virtual cams fully release before the next mode.
            time.sleep(0.2)
            # Several unique modes opened but stayed black → phone/app likely
            # not streaming. Still try None/-1 once, then stop to avoid wedging.
            if black_streak >= 3 and isinstance(attempt_dcap, int) and attempt_dcap >= 0:
                for fb in attempts[i + 1 :]:
                    if fb is not None and fb != -1:
                        continue
                    try:
                        reader = DShowCaptureReader(
                            int(camera_index),
                            width,
                            height,
                            fps,
                            use_dshowcapture=True,
                            dcap=fb,
                        )
                        if self._accept_reader(reader, fb):
                            print(
                                f'Camera OK: "{self.name}" dcap={fb} '
                                f'{self.width}x{self.height} (non-black frame verified)'
                            )
                            return
                        tried.append(f'{fb}:black')
                    except Exception as exc:
                        last_err = exc
                        tried.append(f'{fb}:err')
                    time.sleep(0.2)
                break

        hint = ''
        if any(t.endswith(':black') for t in tried):
            hint = (
                ' — device opened but frames stayed black; '
                'check DroidCam phone connection / start the app stream'
            )
        raise RuntimeError(
            f'DirectShow open failed for index {camera_index} '
            f'(tried {tried}; last={last_err}){hint}'
        )

    def _open_opencv(self, camera_index: int, width: int, height: int, fps: int) -> None:
        ensure_com()
        import time

        # After an aggressive DirectShow probe, DroidCam needs a beat to reopen.
        time.sleep(0.35)

        # DSHOW first so CAP_PROP_AUDIO_STREAM is more likely honored; MSMF still gets params.
        backends = []
        if os.name == 'nt':
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        last_err: Exception | None = None
        for be in backends:
            cap = None
            try:
                cap = _open_cv_index(camera_index, be)
                if not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                for w, h in ((width, height), (1280, 720), (640, 480), (0, 0)):
                    if w and h:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    if fps:
                        cap.set(cv2.CAP_PROP_FPS, fps)
                    best = None
                    for _ in range(25):
                        ok, frame = cap.read()
                        if ok and _frame_ok(frame):
                            best = frame
                            break
                        time.sleep(0.05)
                    if best is not None:
                        self._cap = cap
                        if self.name.startswith('Camera '):
                            self.name = f'Camera {camera_index}'
                        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or best.shape[1])
                        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or best.shape[0])
                        print(
                            f'Camera OK: "{self.name}" opencv be={be} '
                            f'{self.width}x{self.height} (non-black frame verified)'
                        )
                        return
                cap.release()
            except Exception as exc:
                last_err = exc
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        raise RuntimeError(f'OpenCV could not open camera {camera_index}: {last_err}')

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            if self._reader is not None:
                ok, frame = self._reader.read()
                if ok and frame is not None and frame.size > 0:
                    self.last_health = 'ok' if _frame_ok(frame) else 'black'
                    self.last_error = None
                    return True, frame
                self.last_health = 'stalled'
                return False, None
            if self._cap is not None:
                ok, frame = self._cap.read()
                if ok and frame is not None and frame.size > 0:
                    self.last_health = 'ok' if _frame_ok(frame) else 'black'
                    self.last_error = None
                    return True, frame
                self.last_health = 'stalled'
                return False, None
        except Exception as exc:
            self.last_health = 'error'
            self.last_error = str(exc)
            return False, None
        self.last_health = 'closed'
        return False, None

    def release(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
