"""Live tracking must drop stale camera / UDP frames instead of queuing delay."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from backend.ifm_cam import recv_latest
from backend.osf_cam import OsfCam
from backend import osf_cam as osf_cam_mod


def test_recv_latest_keeps_newest_datagram() -> None:
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        recv.bind(("127.0.0.1", 0))
        recv.settimeout(1.0)
        addr = ("127.0.0.1", int(recv.getsockname()[1]))
        for payload in (b"one", b"two", b"three"):
            send.sendto(payload, addr)
        got, _peer = recv_latest(recv)
        assert got == b"three"
    finally:
        recv.close()
        send.close()


def test_tracker_is_built_before_the_camera_opens(monkeypatch) -> None:
    order: list[str] = []

    class FakeTracker:
        def close(self) -> None:
            return None

    def make_tracker(w: int, h: int) -> FakeTracker:
        order.append(f"tracker:{w}x{h}")
        return FakeTracker()

    def open_camera(index: int):
        order.append(f"open:{index}")
        return None

    monkeypatch.setattr(osf_cam_mod, "_make_tracker", make_tracker)
    monkeypatch.setattr(osf_cam_mod, "_open_camera", open_camera)
    cam = OsfCam()
    try:
        cam.start(index=3)
    except RuntimeError:
        pass
    assert order == ["tracker:640x480", "open:3"]


def test_osf_processes_latest_frame_not_backlog(monkeypatch) -> None:
    produced = {"n": 0}
    processed: list[int] = []

    class FakeCap:
        def read(self):
            produced["n"] += 1
            n = produced["n"]
            time.sleep(0.004)
            return True, np.full((12, 12, 3), n % 255, dtype=np.uint8)

        def release(self) -> None:
            return None

    class FakeTracker:
        def predict(self, frame):
            processed.append(int(frame[0, 0, 0]))
            time.sleep(0.03)
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr(osf_cam_mod, "_make_tracker", lambda w, h: FakeTracker())
    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", lambda img: b"x")

    cam = OsfCam()
    cam._cap = FakeCap()
    cam._running = True
    thread = threading.Thread(target=cam._loop, args=(lambda snap: None,), daemon=True)
    cam._thread = thread
    thread.start()
    time.sleep(0.22)
    cam.stop()
    thread.join(timeout=2.0)
    assert produced["n"] >= 8
    assert len(processed) >= 2
    assert produced["n"] > len(processed) * 2


def test_ifm_processes_latest_packet_not_backlog(monkeypatch) -> None:
    from backend import ifm_cam as ifm_cam_mod
    from backend.ifm_cam import IfmCam

    monkeypatch.setattr(ifm_cam_mod, "_encode_jpeg", lambda img: b"x")

    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    processed: list[float] = []
    try:
        listen.bind(("127.0.0.1", 0))
        listen.settimeout(0.05)
        port = int(listen.getsockname()[1])
        addr = ("127.0.0.1", port)
        cam = IfmCam()
        cam._sock = listen
        cam._running = True
        cam._started = time.perf_counter()

        def on_frame(snap) -> None:
            if not snap.faces:
                return
            processed.append(float((snap.head or {}).get("yaw") or 0.0))
            time.sleep(0.04)

        thread = threading.Thread(target=cam._loop, args=(on_frame,), daemon=True)
        cam._thread = thread
        thread.start()
        for index in range(1, 25):
            send.sendto(f"jawOpen-0|=head#0,{index},0".encode("utf-8"), addr)
            time.sleep(0.004)
        time.sleep(0.12)
        cam.stop()
        thread.join(timeout=2.0)
        assert len(processed) >= 1
        assert len(processed) < 18
        assert processed[-1] >= 18
    finally:
        send.close()
        try:
            listen.close()
        except OSError:
            pass


def _settle_s(smooth: float, fps: float, move: float = 0.5, turn_s: float = 0.0) -> float:
    """Seconds for the head ease to cover 90 % of a turn (radians) the head
    makes in ``turn_s`` seconds (0 = at once)."""
    from backend.ease import HeadEase

    ease = HeadEase()
    ease.step((0.0,), (1.0,), smooth, 0.0)
    for i in range(1, int(fps * 5)):
        now = i / fps
        target = move * min(now / turn_s, 1.0) if turn_s > 0.0 else move
        if float(ease.step((target,), (1.0,), smooth, now)[0]) >= 0.9 * move:
            return now
    return 5.0


def test_stronger_smooth_eases_a_move_longer() -> None:
    """The old blend let go on any move: Smooth only touched a still head."""
    light = _settle_s(0.25, 30.0)
    mid = _settle_s(0.5, 30.0)
    strong = _settle_s(1.0, 30.0)
    assert light < mid < strong
    assert strong > 0.3
    # Smooth 0 is the raw head.
    assert _settle_s(0.0, 30.0) <= 1.0 / 30.0


def test_head_ease_is_timed_in_seconds_not_frames() -> None:
    """One Smooth feels the same on a 12 fps webcam and a 60 fps iPhone."""
    for smooth in (0.5, 1.0):
        # A brisk turn, 0.3 s end to end; 12 fps lands only every 83 ms.
        slow = _settle_s(smooth, 12.0, turn_s=0.3)
        fast = _settle_s(smooth, 60.0, turn_s=0.3)
        assert abs(slow - fast) <= 1.0 / 12.0 + 0.2 * max(slow, fast)


def test_head_ease_holds_jitter_and_keeps_easing_a_move() -> None:
    import numpy as np

    from backend.ease import HeadEase

    rng = np.random.default_rng(0)
    ease = HeadEase()
    out = [float(ease.step((v,), (1.0,), 0.5, i / 30.0)[0]) for i, v in enumerate(rng.normal(0, 0.004, 90))]
    # Tracker jitter held still: well under half of it gets through.
    assert np.std(np.diff(out[30:])) < 0.5 * 0.004 * np.sqrt(2.0)
    # A steady 12 fps turn: every frame trails the reading (still easing,
    # not snapped to it), then settles on it once the head stops.
    ease = HeadEase()
    ease.step((0.0,), (1.0,), 0.5, 0.0)
    for i in range(1, 7):
        target = 0.08 * i
        assert float(ease.step((target,), (1.0,), 0.5, i / 12.0)[0]) < target - 0.005
    for i in range(7, 40):
        got = float(ease.step((0.48,), (1.0,), 0.5, i / 12.0)[0])
    assert abs(got - 0.48) < 0.005


def test_rig_eases_the_turn_but_not_the_mouth(monkeypatch) -> None:
    """Smooth eases the head in the rig; the mesh it is handed (mouth,
    brows) goes through at once, so a strong Smooth does not blur lip sync."""
    import math
    from types import SimpleNamespace

    import numpy as np

    from backend import rig as rig_mod
    from backend.feel import feel
    from backend.rig import FaceRig

    clock = {"t": 0.0}
    monkeypatch.setattr(rig_mod, "time", SimpleNamespace(perf_counter=lambda: clock["t"]))
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[21] = [50.0, 70.0, 1.0]
    rest[25] = [50.0, 72.0, 1.0]
    pose = {"cx": 200.0, "cy": 200.0, "bx": 200.0, "by": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    still = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    turned = {"pitch": 0.0, "yaw": 20.0, "roll": 0.0}

    def yaw_after(smooth: float, frames: int) -> float:
        feel.update({"smoothing": smooth, "max_yaw": 1.0})
        rig = FaceRig()
        clock["t"] = 0.0
        rig.apply(rest, rest, still, pose)
        for i in range(1, frames + 1):
            clock["t"] = i / 30.0
            rig.apply(rest, rest, turned, pose)
        return math.degrees(rig.turn()["yaw"])

    assert abs(yaw_after(0.0, 1) - 20.0) < 0.01
    assert yaw_after(1.0, 1) < 0.25 * 20.0
    assert yaw_after(0.5, 6) > yaw_after(1.0, 6)
    assert abs(yaw_after(1.0, 90) - 20.0) < 0.5

    feel.update({"smoothing": 1.0})
    rig = FaceRig()
    clock["t"] = 0.0
    rig.apply(rest, rest, still, pose)
    opened = rest.copy()
    opened[25, 1] = 84.0
    clock["t"] = 1.0 / 30.0
    out = rig.apply(opened, rest, still, pose)
    assert out is not None
    assert abs(float(out[25, 1] - out[21, 1]) - 14.0) < 0.01
