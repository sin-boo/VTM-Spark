"""Record raw iFacialMocap packets while a voice guides the moves.

    python -m tools.record_ifm              # guided, about a minute
    python -m tools.record_ifm --free 60    # 60 s, no prompts

Writes output/captures/ifm_raw_<time>.jsonl: a header, one line per guide
step, and one line per datagram (receive time, step, sender, raw text), so
the session can be replayed through the tracker offline. Only one program
can listen to the iPhone: Track Lab must not be on iFacialMocap meanwhile.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.ifm import DEFAULT_PORT, HANDSHAKE, handshake_targets, parse_packet
from backend.ifm_cam import _quiet_udp
from tools.guide import Guide, Voice

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "output" / "captures"
IFM_SETTINGS = ROOT / "output" / "ifm.json"
# Knock until the phone streams. Knocking while it streams makes it reopen
# its sender and drop frames.
PING_SEC = 1.0
QUIET_SEC = 1.5
WAIT_SEC = 60.0


def saved_phone() -> dict[str, Any]:
    """Phone address and port Track Lab last used."""
    try:
        data = json.loads(IFM_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def listen(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
        sock.bind(("", int(port)))
    except OSError as exc:
        sock.close()
        raise SystemExit(
            f"UDP {port} is taken, most likely by Track Lab listening to the iPhone. "
            "Switch Track Lab's input to Camera (or close VTM Spark) and run this again."
        ) from exc
    _quiet_udp(sock)
    sock.settimeout(0.02)
    return sock


def summarize(rows: list[dict[str, Any]]) -> list[str]:
    """Median head per step, from rest (or from the whole take when there are
    no steps), in iFacialMocap's own signs: pitch / yaw / roll in degrees and
    the head's position in cm, as the phone sent them."""
    by_step: dict[str, list[tuple[float, ...]]] = {}
    for row in rows:
        packet = parse_packet(row["raw"])
        if packet is None or not packet.has_head:
            continue
        head = packet.head
        pos = packet.position
        by_step.setdefault(row["step"], []).append(
            (
                float(head["pitch"]),
                float(head["yaw"]),
                float(head["roll"]),
                100.0 * float(pos.get("x", 0.0)),
                100.0 * float(pos.get("y", 0.0)),
                100.0 * float(pos.get("z", 0.0)),
            )
        )
    if not by_step:
        return ["No head data."]
    rest = by_step.get("rest") or by_step.get("center")
    where = "from rest"
    if not rest:
        rest = [value for values in by_step.values() for value in values]
        where = "from the median"
    zero = [statistics.median(axis) for axis in zip(*rest)]
    lines = [
        f"{'step':12s} {'pitch':>7s} {'yaw':>7s} {'roll':>7s} {'x cm':>7s} {'y cm':>7s} {'z cm':>7s}"
        f"   raw, {where}; at rest pitch {zero[0]:+.1f} yaw {zero[1]:+.1f} roll {zero[2]:+.1f}"
    ]
    for step, values in by_step.items():
        med = [statistics.median(axis) - z for axis, z in zip(zip(*values), zero)]
        cells = " ".join(f"{value:+7.1f}" for value in med)
        lines.append(f"{step:12s} {cells}   ({len(values)} packets)")
    return lines


def record(out: Path, *, phone: str, port: int, free: float, voice_on: bool) -> int:
    saved = saved_phone()
    port = int(port or saved.get("port") or DEFAULT_PORT)
    sock = listen(port)
    host = phone or str(saved.get("host") or "")
    targets = handshake_targets(host, str(saved.get("last_peer") or ""), port)
    voice = Voice(voice_on and free <= 0)
    guide = None if free > 0 else Guide(voice=voice)
    length = free if free > 0 else guide.total  # type: ignore[union-attr]
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    print(f"Listening on UDP {port}; asking {', '.join(ip for ip, _ in targets)} to stream.", flush=True)
    print("Open iFacialMocap on the iPhone (same Wi-Fi).", flush=True)
    try:
        with out.open("w", encoding="utf-8") as fh:
            header = {
                "kind": "header",
                "tool": "record_ifm",
                "started": datetime.now().isoformat(timespec="seconds"),
                "port": port,
                "mirror": saved.get("mirror"),
                "guided": guide is not None,
                "steps": [list(step) for step in guide.steps] if guide is not None else [],
            }
            fh.write(json.dumps(header) + "\n")
            waited = time.perf_counter()
            last_knock = 0.0
            last_packet = 0.0
            start = 0.0
            sender = ""
            step = "free"
            while True:
                now = time.perf_counter()
                if now - last_packet > QUIET_SEC and now - last_knock > PING_SEC:
                    for target in targets:
                        try:
                            sock.sendto(HANDSHAKE.encode("utf-8"), target)
                        except OSError:
                            continue
                    last_knock = now
                if not start and now - waited > WAIT_SEC:
                    print("No packets from iFacialMocap. Is the app open on the same Wi-Fi?", flush=True)
                    return 1
                if start:
                    t = now - start
                    if guide is not None:
                        tag, changed, done = guide.tick(t)
                        if changed:
                            step = tag
                            fh.write(json.dumps({"kind": "step", "t": round(t, 4), "step": tag}) + "\n")
                        if done:
                            break
                    elif t >= length:
                        break
                try:
                    payload, addr = sock.recvfrom(65535)
                except (TimeoutError, socket.timeout):
                    continue
                except OSError:
                    continue
                ip = str(addr[0])
                if sender and ip != sender:
                    continue
                text = payload.decode("utf-8", errors="replace")
                if parse_packet(payload) is None:
                    continue
                now = time.perf_counter()
                last_packet = now
                if not start:
                    sender = ip
                    start = now
                    print(f"Receiving from {ip}. Recording {length:.0f} s.", flush=True)
                    if guide is not None:
                        step, _changed, _done = guide.tick(0.0)
                        fh.write(json.dumps({"kind": "step", "t": 0.0, "step": step}) + "\n")
                row = {"t": round(now - start, 4), "step": step, "ip": ip, "raw": text}
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
    finally:
        sock.close()
        voice.close()
    span = rows[-1]["t"] - rows[0]["t"] if len(rows) > 1 else 0.0
    rate = (len(rows) - 1) / span if span > 0 else 0.0
    print(f"Saved {len(rows)} packets ({rate:.0f} per second) to {out}", flush=True)
    for line in summarize(rows):
        print(line, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path, default=CAPTURES / f"ifm_raw_{stamp}.jsonl")
    parser.add_argument("--phone", default="", help="iPhone IP (default: the one Track Lab last used)")
    parser.add_argument("--port", type=int, default=0, help=f"UDP port (default {DEFAULT_PORT})")
    parser.add_argument("--free", type=float, default=0.0, help="record this many seconds, no prompts")
    parser.add_argument("--quiet", action="store_true", help="print the prompts, do not speak them")
    args = parser.parse_args()
    return record(args.out, phone=args.phone, port=args.port, free=args.free, voice_on=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
