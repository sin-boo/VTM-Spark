"""Listen for iFacialMocap UDP and emit ARKit-driven frames into the live mixer."""

from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from .feel import feel
from .calibrate import calibrator
from .ifm import (
    DEFAULT_PORT,
    HANDSHAKE,
    IFM_H,
    IFM_W,
    IfmFace,
    IfmPacket,
    apply_shapes,
    blink_of,
    brow_of,
    handshake_targets,
    head_of,
    hold_packet,
    iris_of,
    keep_sender,
    lan_ipv4s,
    look_of,
    parse_packet,
    pose_of,
    rest_landmarks,
    sweep_hosts,
    weights_from_arkit,
)
from .mouth_bits import bits as mouth_bits
from .eye_bits import bits as eye_bits
from .osf_cam import OsfFrame, _draw_lid_mids, _encode_jpeg, _smooth, _stamp_id
from .presets import empty_weights
from .retarget import FACE_TRACK
from .rig import project_head
from .visemes import apply_calibrated_rest, mouth_features

_PING_SEC = 2.0
_STALE_SEC = 0.8
_WAIT_HINT_SEC = 6.0
_SIO_UDP_CONNRESET = 0x9800000C
_PIP_REST = rest_landmarks()


def _ingest_ifm_rest(
    pts: np.ndarray,
    head: dict[str, float],
    blink: dict[str, float],
    pose: dict[str, float],
) -> None:
    """Finish Set Rest on the phone path. Webcam rest already goes through visemes."""
    if not calibrator.capturing:
        return
    feat = mouth_features(IfmFace(pts, head, blink), pose)
    if calibrator.ingest(feat) == "rest":
        apply_calibrated_rest()


def _quiet_udp(sock: socket.socket) -> None:
    """Windows otherwise treats ICMP port-unreachable as a socket reset."""
    if sys.platform != "win32":
        return
    try:
        # CPython's socket.ioctl only knows a few codes and raises
        # ValueError for this one; fall back to WSAIoctl through ctypes.
        sock.ioctl(_SIO_UDP_CONNRESET, struct.pack("I", 0))
        return
    except (OSError, ValueError):
        pass
    try:
        import ctypes

        ws2 = ctypes.windll.ws2_32
        flag = ctypes.c_ulong(0)
        out = ctypes.c_ulong(0)
        ws2.WSAIoctl(
            ctypes.c_void_p(sock.fileno()),
            ctypes.c_ulong(_SIO_UDP_CONNRESET),
            ctypes.byref(flag),
            ctypes.sizeof(flag),
            None,
            0,
            ctypes.byref(out),
            None,
            None,
        )
    except Exception:
        pass


def recv_latest(sock: socket.socket, bufsize: int = 8192) -> tuple[bytes, tuple]:
    """One datagram, then drop anything that piled up while tracking ran."""
    payload, addr = sock.recvfrom(bufsize)
    timeout = sock.gettimeout()
    try:
        sock.settimeout(0.0)
        while True:
            payload, addr = sock.recvfrom(bufsize)
    except (TimeoutError, socket.timeout, BlockingIOError, InterruptedError):
        pass
    except OSError:
        pass
    finally:
        try:
            sock.settimeout(timeout)
        except OSError:
            pass
    return payload, addr


class _LatestDatagram:
    """Overwrite mailbox so a slow mixer never replays buffered iPhone packets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: tuple[bytes, tuple] | None = None
        self._has = threading.Event()

    def put(self, item: tuple[bytes, tuple]) -> None:
        with self._lock:
            self._item = item
            self._has.set()

    def take(self, timeout: float | None = None) -> tuple[bytes, tuple] | None:
        if not self._has.wait(timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._has.clear()
        return item


def _unwrap_deg(prev: float, now: float) -> float:
    delta = (float(now) - float(prev) + 180.0) % 360.0 - 180.0
    return float(prev) + delta


def _mix_head(
    prev: dict[str, float] | None, nxt: dict[str, float], alpha: float
) -> dict[str, float]:
    if prev is None:
        return {
            "pitch": float(nxt.get("pitch", 0.0)),
            "yaw": float(nxt.get("yaw", 0.0)),
            "roll": float(nxt.get("roll", 0.0)),
        }
    out = dict(nxt)
    for key in ("pitch", "yaw", "roll"):
        old = float(prev.get(key, 0.0))
        target = _unwrap_deg(old, float(nxt.get(key, 0.0)))
        delta = target - old
        if abs(delta) < 0.35:
            out[key] = old
        else:
            out[key] = old + alpha * delta
    return out


class IfmCam:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.host = ""
        self.last_peer = ""
        self.port = DEFAULT_PORT
        self.latest = OsfFrame()
        self.peer = ""
        self.fps = 0.0
        self.receiving = False
        self._started = 0.0
        self._head_s: dict[str, float] | None = None
        self._lock_ip = ""
        self._held_pkt: IfmPacket | None = None
        self._primary = ""

    @property
    def running(self) -> bool:
        return self._running

    def payload(self) -> dict[str, object]:
        local = lan_ipv4s()
        with self._lock:
            waiting = self._running and not self.receiving
            elapsed = time.perf_counter() - self._started if self._started else 0.0
            primary = local[0] if local else ""
            if self.receiving and self.peer:
                hint = f"Live from {self.peer}"
            elif waiting and elapsed >= _WAIT_HINT_SEC:
                dest = f"{primary}:{self.port}" if primary else f"UDP {self.port}"
                hint = (
                    f"No packets yet. Same Wi-Fi, allow Python on Private networks, "
                    f"iPhone destination {dest}."
                )
            elif self._running:
                hint = "Waiting for iFacialMocap on this LAN"
            elif primary:
                hint = f"Listen. On the iPhone, destination is {primary}."
            else:
                hint = "Listen. Plug into the same Wi-Fi as the iPhone."
            return {
                "host": self.host,
                "last_peer": self.last_peer,
                "port": int(self.port),
                "listening": self._running,
                "receiving": self.receiving,
                "fps": round(float(self.fps), 1),
                "peer": self.peer,
                "local": local,
                "primary": primary,
                "hint": hint,
            }

    def configure(self, host: str | None = None, port: int | None = None) -> None:
        with self._lock:
            if host is not None:
                text = str(host).strip()
                # This PC's own address is not the phone. It ends up saved
                # when the "destination" hint gets typed back into the box.
                self.host = "" if text and text in lan_ipv4s() else text
            if port is not None:
                self.port = int(port) if int(port) > 0 else DEFAULT_PORT

    def start(
        self,
        on_frame: Callable[[OsfFrame], None] | None = None,
        host: str = "",
        port: int = DEFAULT_PORT,
    ) -> None:
        if self._running:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("iFacialMocap is still stopping")
        self.configure(host, port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # No SO_REUSEADDR: on Windows it lets a stale listener share the
        # port silently, splitting the phone's datagrams between processes.
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if sys.platform == "win32" and exclusive is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            except OSError:
                pass
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        try:
            sock.bind(("", int(self.port)))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"Could not listen on UDP {self.port}. Close the app using that port."
            ) from exc
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        except OSError:
            pass
        try:
            _quiet_udp(sock)
            sock.settimeout(0.05)
        except Exception:
            sock.close()
            raise
        self._sock = sock
        self._running = True
        self.latest = OsfFrame()
        self.peer = ""
        self.fps = 0.0
        self.receiving = False
        self._started = time.perf_counter()
        self._head_s = None
        self._lock_ip = ""
        self._held_pkt = None
        local = lan_ipv4s()
        self._primary = local[0] if local else ""
        self._thread = threading.Thread(
            target=self._loop,
            args=(on_frame,),
            name="ifm-cam",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        recv = self._recv_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        if recv is not None and recv.is_alive():
            recv.join(timeout=0.2)
        self._release()
        if recv is not None and recv.is_alive():
            recv.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            thread.join(timeout=8.0)
        if recv is None or not recv.is_alive():
            self._recv_thread = None
        if thread is None or not thread.is_alive():
            self._thread = None

    def _release(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    def _ping(self, sock: socket.socket, now: float, last: float) -> float:
        # The handshake is a "start streaming" command. Re-sending it while
        # the phone is already streaming makes it reopen its sender (new
        # source port) and drop frames. Only knock while nothing arrives.
        if self.receiving:
            return now
        if now - last < _PING_SEC:
            return last
        payload = HANDSHAKE.encode("utf-8")
        for ip, port in handshake_targets(self.host, self.last_peer, self.port):
            try:
                sock.sendto(payload, (ip, port))
            except OSError:
                continue
        return now

    def _sweep(self, sock: socket.socket) -> None:
        payload = HANDSHAKE.encode("utf-8")
        skip = set(lan_ipv4s())
        if self.host:
            skip.add(self.host)
        if self.last_peer:
            skip.add(self.last_peer)
        port = int(self.port)
        for ip in sweep_hosts(skip):
            try:
                sock.sendto(payload, (ip, port))
            except OSError:
                continue

    def _recv_loop(self, sock: socket.socket, pending: _LatestDatagram) -> None:
        """Keep eating iPhone datagrams so UDP never queues a delay."""
        last_ping = 0.0
        swept = False
        try:
            while self._running:
                now = time.perf_counter()
                last_ping = self._ping(sock, now, last_ping)
                if (
                    not self.receiving
                    and not swept
                    and now - self._started >= _WAIT_HINT_SEC
                ):
                    self._sweep(sock)
                    swept = True
                try:
                    pending.put(recv_latest(sock))
                except (TimeoutError, socket.timeout):
                    continue
                except OSError:
                    if not self._running:
                        break
                    continue
        finally:
            self._recv_thread = None

    def _loop(self, on_frame: Callable[[OsfFrame], None] | None) -> None:
        sock = self._sock
        smoothed = empty_weights()
        last_pkt = 0.0
        last_idle = 0.0
        hits = 0
        fps_t = time.perf_counter()
        pending = _LatestDatagram()
        try:
            if sock is None:
                raise RuntimeError("iFacialMocap socket closed")
            recv = threading.Thread(
                target=self._recv_loop,
                args=(sock, pending),
                name="ifm-recv",
                daemon=True,
            )
            self._recv_thread = recv
            recv.start()
            while self._running:
                item = pending.take(timeout=0.05)
                now = time.perf_counter()
                if item is None:
                    self._idle(on_frame, now, last_pkt, last_idle)
                    if now - last_idle >= 0.45:
                        last_idle = now
                    continue
                payload, addr = item
                ip = str(addr[0])
                if not keep_sender(self._lock_ip, ip, live=self.receiving):
                    continue
                packet = parse_packet(payload)
                if packet is None:
                    continue
                if self._held_pkt is None and not packet.has_head and len(packet.shapes) < 20:
                    continue
                packet = hold_packet(self._held_pkt, packet)
                self._held_pkt = packet
                self._lock_ip = ip
                started = time.perf_counter()
                weights = weights_from_arkit(packet)
                smoothed = _smooth(smoothed, weights, feel.alpha())
                head_a = min(max(feel.alpha(), 0.12), 0.22)
                head = _mix_head(self._head_s, head_of(packet), head_a)
                self._head_s = head
                pose = pose_of(packet)
                pose["tilt"] = float(head.get("roll", 0.0))
                look = look_of(packet)
                pip_pts = apply_shapes(_PIP_REST, packet)
                blink = blink_of(packet)
                _ingest_ifm_rest(pip_pts, head, blink, pose)
                hits += 1
                if started - fps_t >= 1.0:
                    self.fps = hits / (started - fps_t)
                    hits = 0
                    fps_t = started
                last_pkt = started
                self.receiving = True
                self.peer = ip
                self.last_peer = ip
                snap = OsfFrame(
                    weights=smoothed,
                    head=dict(head),
                    blink=blink,
                    pose=pose,
                    faces=1,
                    ms=(time.perf_counter() - started) * 1000.0,
                    camera_jpeg=_encode_jpeg(
                        _draw_ifm(
                            pip_pts,
                            True,
                            {"peer": ip, "fps": self.fps, "primary": self._primary},
                            head,
                            look,
                        )
                    ),
                    look=look,
                    source="ifm",
                    brow=brow_of(packet),
                )
                self.latest = snap
                if on_frame is not None:
                    on_frame(snap)
        except Exception as exc:
            snap = OsfFrame(error=str(exc), source="ifm")
            self.latest = snap
            if on_frame is not None:
                on_frame(snap)
            self._running = False
        finally:
            self.receiving = False
            self._release()
            self._thread = None

    def _idle(
        self,
        on_frame: Callable[[OsfFrame], None] | None,
        now: float,
        last_pkt: float,
        last_idle: float,
    ) -> None:
        live = last_pkt > 0.0 and (now - last_pkt) < _STALE_SEC
        self.receiving = live
        if live or now - last_idle < 0.45:
            return
        self.fps = 0.0
        info = self.payload()
        self._primary = str(info.get("primary") or "")
        jpeg = _encode_jpeg(_draw_ifm(None, False, info))
        snap = OsfFrame(camera_jpeg=jpeg, faces=0, source="ifm")
        self.latest = snap
        if on_frame is not None:
            on_frame(snap)


def _draw_ifm(
    pts: np.ndarray | None,
    live: bool,
    info: dict[str, object],
    head: dict[str, float] | None = None,
    look: dict[str, float] | None = None,
) -> np.ndarray:
    vis = np.full((IFM_H, IFM_W, 3), (18, 16, 12), dtype=np.uint8)
    cv2.rectangle(vis, (16, 16), (IFM_W - 16, IFM_H - 16), (58, 52, 42), 1, cv2.LINE_AA)
    title = "iFacialMocap"
    cv2.putText(
        vis, title, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 162, 201), 1, cv2.LINE_AA
    )
    status = "receiving" if live else "listening"
    peer = str(info.get("peer") or "")
    fps = float(info.get("fps") or 0.0)
    line = f"{status}  ·  {fps:.0f} fps"
    if peer:
        line += f"  ·  {peer}"
    cv2.putText(
        vis, line, (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 214, 232), 1, cv2.LINE_AA
    )
    hint = str(info.get("primary") or "iFacialMocap UDP")
    if hint and ":" not in hint:
        hint = f"this PC  {hint}"
    cv2.putText(
        vis, hint, (28, IFM_H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (122, 116, 104), 1, cv2.LINE_AA
    )
    show_ids = feel.show_ids()
    if pts is None or not (feel.show_face() or show_ids):
        return vis
    pts_arr = np.asarray(pts, dtype=np.float32)
    xs = pts_arr[:, 0]
    ys = pts_arr[:, 1]
    jaw = xs[:17] if len(xs) > 16 else xs
    radius = max(float(np.ptp(jaw)) * 1.05, 1.0)
    xs, ys = project_head(xs, ys, head, radius)
    cx, cy, scale = IFM_W * 0.5, IFM_H * 0.52, 160.0
    pix = np.empty((len(pts_arr), 3), dtype=np.float32)
    pix[:, 0] = cx + xs * scale
    pix[:, 1] = cy + ys * scale
    pix[:, 2] = 1.0
    ink = (236, 228, 208)
    font = 0.33
    for i in range(min(66, len(pts_arr))):
        if not show_ids and 48 <= i <= 65 and not mouth_bits.on(i):
            continue
        px, py = int(round(float(pix[i, 0]))), int(round(float(pix[i, 1])))
        if px < 8 or py < 8 or px >= IFM_W - 8 or py >= IFM_H - 8:
            continue
        used_mouth = 48 <= i <= 65 and mouth_bits.on(i)
        used_eye = 36 <= i <= 47 and eye_bits.on(i)
        used_face = i in FACE_TRACK and not (36 <= i <= 47)
        if used_mouth:
            color = (180, 80, 255)
        elif used_eye:
            color = (80, 230, 160)
        elif used_face:
            color = (80, 200, 255)
        else:
            color = (160, 168, 176)
        cv2.circle(vis, (px, py), 3, (8, 10, 14), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 2, color, -1, cv2.LINE_AA)
        if show_ids:
            _stamp_id(vis, px, py, str(i), ink, font)
    _draw_lid_mids(vis, pix, 1, 2, font, show_ids=show_ids)
    if isinstance(look, dict):
        iris = iris_of(pts_arr, look)
        ixs, iys = project_head(iris[:, 0], iris[:, 1], head, radius)
        color = (40, 180, 255)
        for i, (ix, iy) in enumerate(zip(ixs, iys)):
            px = int(round(cx + float(ix) * scale))
            py = int(round(cy + float(iy) * scale))
            if px < 8 or py < 8 or px >= IFM_W - 8 or py >= IFM_H - 8:
                continue
            cv2.circle(vis, (px, py), 5, (8, 10, 14), -1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 4, color, 1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 2, color, -1, cv2.LINE_AA)
            if show_ids:
                _stamp_id(vis, px, py, f"i{i}", color, font, pad=8)
    return vis
