"""Stand-in for a game running next to the desk: holds VRAM, keeps the GPU and CPU busy.

Run it by hand next to the app, or let ``backend.tools.bench_game_load`` drive it::

    python -m backend.tools.stress_game --card-gb 8            # this GPU acts like an 8 GB card
    python -m backend.tools.stress_game --card-gb 8 --game-gb 3 --gpu-busy 0.6 --fps 60
    python -m backend.tools.stress_game --leave-gb 1.5 --cpu-threads 4 --cpu-busy 0.8
    python -m backend.tools.stress_game --vram-gb 6 --vram-mode force   # past free VRAM: WDDM spill

VRAM: ``--card-gb C`` holds ``total - C`` so the rest of the PC (desktop, browser, the desk)
lives inside C, as on that card; ``--game-gb G`` adds what the game itself would hold;
``--leave-gb L`` holds until L is free (L < 0: |L| more than is free); ``--vram-gb N`` holds N. ``--vram-mode fit`` (default)
stops at free VRAM minus ``--margin-mb``; ``force`` keeps asking, so the NVIDIA driver either
refuses (sysmem fallback off) or places/evicts allocations in shared system memory (the
default "Driver default" policy) -- the slow path a real game drives a small card into. Every
held chunk is touched each frame so WDDM keeps it resident, like a game's textures.

GPU: ``--gpu-busy D`` runs a fixed fp16 matmul load sized to D of a ``--fps`` frame, then
sleeps out the frame. The work per frame is fixed, so when the desk competes the frame gets
longer -- the "game slows too" half of the report shows up as this tool's frame time. This is
CUDA, not D3D, but on Windows both are GPU contexts the WDDM scheduler time-slices the same
way (and with HAGS the GPU's own scheduler does), which is the contention that matters here.

CPU: ``--cpu-threads K --cpu-busy D`` spins K processes at duty D. ``--cpus 0-5`` pins this
tool (and its workers) to those logical CPUs; pin the bench to the same set to act out a
6-thread CPU.

Machine-readable lines start with ``STRESS `` + JSON (ready / stats / exit). Stops on Ctrl+C,
``--duration``, once ``--stop-file`` exists, or when ``--parent-pid`` exits.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from typing import Any

GIB = 1024**3
MIB = 1024**2


def parse_cpus(spec: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def pin_cpus(cpus: list[int]) -> None:
    if not cpus:
        return
    try:
        import psutil

        psutil.Process().cpu_affinity(cpus)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[stress] could not pin to CPUs {cpus}: {exc}", flush=True)


def emit(event: str, **fields: Any) -> None:
    print("STRESS " + json.dumps({"event": event, **fields}), flush=True)


def vram_target(total: int, free: int, args: argparse.Namespace) -> int:
    """Bytes to hold, from the --card-gb / --leave-gb / --vram-gb / --game-gb flags."""
    target = 0.0
    if args.card_gb is not None:
        target += max(0.0, total - args.card_gb * GIB)
    elif args.leave_gb is not None:
        target += max(0.0, free - args.leave_gb * GIB)
    if args.vram_gb is not None:
        target += args.vram_gb * GIB
    if args.game_gb:
        target += args.game_gb * GIB
    return int(target)


def _cpu_worker(stop: Any, busy: float, period: float, cpus: list[int]) -> None:
    pin_cpus(cpus)
    busy = min(max(busy, 0.0), 1.0)
    x = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        end = t0 + busy * period
        while time.perf_counter() < end:
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        rest = period - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def _watch_parent(pid: int, stop: threading.Event) -> None:
    try:
        import psutil
    except Exception:
        return
    while not stop.is_set():
        if not psutil.pid_exists(pid):
            print(f"[stress] parent {pid} gone, stopping", flush=True)
            stop.set()
            return
        time.sleep(0.5)


def _watch_stop_file(path: str, stop: threading.Event) -> None:
    # A file, not stdin: on Windows a thread blocked reading a stdin pipe stalls the main
    # thread's own I/O (torch import, CUDA init) until the pipe closes.
    while not stop.is_set():
        if os.path.exists(path):
            stop.set()
            return
        time.sleep(0.2)


class CardMemory:
    """(free, total) for the whole card. NVML counts every process, as Task Manager does;
    ``cudaMemGetInfo`` under WDDM leaves part of the other processes out, so a fit based on
    it lands past the card's real VRAM."""

    def __init__(self, torch: Any, device: int) -> None:
        self.torch, self.device, self.nv = torch, device, None
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self.nv = pynvml
        except Exception:
            self.nv = None

    def info(self) -> tuple[int, int]:
        if self.nv is not None:
            m = self.nv.nvmlDeviceGetMemoryInfo(self.handle)
            return int(m.total) - int(m.used), int(m.total)
        free, total = self.torch.cuda.mem_get_info()
        return int(free), int(total)


def grab_vram(
    torch: Any, target: int, args: argparse.Namespace, card: CardMemory
) -> tuple[list[Any], dict[str, Any]]:
    chunk = max(16, int(args.chunk_mb)) * MIB
    margin = max(0, int(args.margin_mb)) * MIB
    chunks: list[Any] = []
    held = 0
    stop_reason = "target"
    while held < target:
        want = min(chunk, target - held)
        if args.vram_mode == "fit":
            free, _total = card.info()
            room = free - margin
            if room < 16 * MIB:
                stop_reason = "free VRAM reached"
                break
            want = min(want, room)
        try:
            t = torch.empty(int(want), dtype=torch.uint8, device="cuda")
            t.zero_()  # commit the pages, not just reserve the range
            torch.cuda.synchronize()
        except Exception as exc:  # torch.OutOfMemoryError / RuntimeError
            stop_reason = f"allocation refused: {str(exc).splitlines()[0][:160]}"
            break
        chunks.append(t)
        held += int(want)
    free, total = card.info()
    info = {
        "target_bytes": int(target),
        "held_bytes": int(held),
        "reached": held >= target,
        "stop_reason": stop_reason,
        "free_after": int(free),
        "total": int(total),
        "chunks": len(chunks),
    }
    return chunks, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    vram = ap.add_argument_group("VRAM")
    vram.add_argument("--card-gb", type=float, default=None, help="act like a card this big (holds total - C)")
    vram.add_argument("--leave-gb", type=float, default=None, help="hold until this much is free (negative: that much past free, with --vram-mode force)")
    vram.add_argument("--vram-gb", type=float, default=None, help="hold this much")
    vram.add_argument("--game-gb", type=float, default=0.0, help="extra VRAM the game itself holds")
    vram.add_argument("--vram-mode", choices=("fit", "force"), default="fit")
    vram.add_argument("--margin-mb", type=int, default=256, help="fit mode: leave this free")
    vram.add_argument("--chunk-mb", type=int, default=256)
    vram.add_argument("--bw-mb", type=float, default=0.0, help="MB of held VRAM read+written per frame")
    gpu = ap.add_argument_group("GPU compute")
    gpu.add_argument("--gpu-busy", type=float, default=0.0, help="0..1 of each frame spent on matmuls")
    gpu.add_argument("--fps", type=float, default=60.0, help="game frame rate the load is paced to")
    gpu.add_argument("--matmul", type=int, default=2048, help="square fp16 matmul size per work item")
    cpu = ap.add_argument_group("CPU")
    cpu.add_argument("--cpu-threads", type=int, default=0)
    cpu.add_argument("--cpu-busy", type=float, default=1.0)
    cpu.add_argument("--cpus", default="", help="pin to these logical CPUs, e.g. 0-5")
    run = ap.add_argument_group("run")
    run.add_argument("--duration", type=float, default=0.0, help="seconds, 0 = until stopped")
    run.add_argument("--parent-pid", type=int, default=0)
    run.add_argument("--stop-file", default="", help="stop once this file exists")
    run.add_argument("--stats-every", type=float, default=1.0)
    run.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)

    if args.card_gb is not None and args.leave_gb is not None:
        ap.error("--card-gb and --leave-gb both set the free VRAM; pick one")

    cpus = parse_cpus(args.cpus)
    pin_cpus(cpus)

    stop = threading.Event()

    def _on_signal(*_a: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_signal)
    if args.parent_pid:
        threading.Thread(target=_watch_parent, args=(args.parent_pid, stop), daemon=True).start()
    if args.stop_file:
        threading.Thread(target=_watch_stop_file, args=(args.stop_file, stop), daemon=True).start()

    workers: list[Any] = []
    cpu_stop = None
    if args.cpu_threads > 0:
        ctx = mp.get_context("spawn")
        cpu_stop = ctx.Event()
        for _ in range(args.cpu_threads):
            p = ctx.Process(
                target=_cpu_worker, args=(cpu_stop, args.cpu_busy, 0.01, cpus), daemon=True
            )
            p.start()
            workers.append(p)

    need_gpu = (
        args.card_gb is not None
        or args.leave_gb is not None
        or args.vram_gb is not None
        or args.game_gb > 0
        or args.gpu_busy > 0
    )
    chunks: list[Any] = []
    info: dict[str, Any] = {}
    torch = None
    try:
        if need_gpu:
            # The desk's expandable_segments would let held chunks be remapped; a game's
            # allocations are plain ones.
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            import torch as _torch

            torch = _torch
            torch.cuda.set_device(args.device)
            torch.zeros(1, device="cuda")
            card = CardMemory(torch, args.device)
            free, total = card.info()
            target = vram_target(total, free, args)
            chunks, info = grab_vram(torch, target, args, card)
            info["free_before"] = int(free)
            info["free_source"] = "nvml" if card.nv is not None else "cudaMemGetInfo"
        emit(
            "ready",
            pid=os.getpid(),
            cpu_threads=args.cpu_threads,
            gpu_busy=args.gpu_busy,
            **info,
        )
        if info and not info.get("reached", True):
            print(
                f"[stress] held {info['held_bytes'] / GIB:.2f} of {info['target_bytes'] / GIB:.2f} GiB "
                f"({info['stop_reason']})",
                flush=True,
            )

        period = 1.0 / max(1.0, args.fps)
        k = 0
        a = b = c = None
        if torch is not None and args.gpu_busy > 0:
            n = int(args.matmul)
            a = torch.randn(n, n, device="cuda", dtype=torch.float16)
            b = torch.randn(n, n, device="cuda", dtype=torch.float16)
            c = torch.empty(n, n, device="cuda", dtype=torch.float16)
            for _ in range(5):
                torch.mm(a, b, out=c)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                torch.mm(a, b, out=c)
            e1.record()
            e1.synchronize()
            per_mm = e0.elapsed_time(e1) / 20.0 / 1000.0
            busy = min(max(args.gpu_busy, 0.0), 1.0)
            k = max(1, round(busy * period / per_mm))
            if busy >= 0.999:
                period = 0.0
            emit("calibrated", matmul_ms=per_mm * 1000.0, per_frame=k, frame_ms=period * 1000.0)

        bw_bytes = int(args.bw_mb * MIB)
        done = torch.cuda.Event(blocking=True) if torch is not None else None
        frames: list[float] = []
        last_stats = time.perf_counter()
        t_start = last_stats
        bw_pos = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            if torch is not None:
                for _ in range(k):
                    torch.mm(a, b, out=c)
                for ch in chunks:  # residency: any touch keeps the allocation in VRAM
                    ch[:4096].add_(1)
                if bw_bytes and chunks:
                    left = bw_bytes
                    while left > 0:
                        ch = chunks[bw_pos % len(chunks)]
                        n_b = min(left, ch.numel())
                        ch[:n_b].add_(1)
                        left -= n_b
                        bw_pos += 1
                done.record()
                done.synchronize()
            dt = time.perf_counter() - t0
            frames.append(dt)
            rest = period - dt
            # time.sleep is a high-resolution timer on Windows (3.11+); Event.wait rounds
            # up to the 15.6 ms system tick and would halve a 60 fps pace.
            if rest > 0:
                time.sleep(rest)
            elif torch is None:
                time.sleep(0.05)
            now = time.perf_counter()
            if now - last_stats >= args.stats_every and frames:
                s = sorted(frames)
                emit(
                    "stats",
                    t=now,
                    frames=len(s),
                    fps=len(s) / (now - last_stats),
                    work_ms_p50=s[len(s) // 2] * 1000.0,
                    work_ms_p95=s[min(len(s) - 1, int(len(s) * 0.95))] * 1000.0,
                )
                frames = []
                last_stats = now
            if args.duration and now - t_start >= args.duration:
                break
    finally:
        if cpu_stop is not None:
            cpu_stop.set()
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        held = sum(int(ch.numel()) for ch in chunks)
        chunks.clear()
        if torch is not None:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        emit("exit", released_bytes=held)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
