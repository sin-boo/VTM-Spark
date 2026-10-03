"""Stream engine speed with a "game" next to it: ms per key, keys/s, VRAM, sysmem spill.

Drives ``backend.engine.StreamEngine`` in this process (no UI, camera or Track Lab) the way
the stream does -- Fast mode, 1 step, the whole-frame CUDA graph at the session's batch --
and runs ``backend.tools.stress_game`` beside it for each scenario::

    python -m backend.tools.bench_game_load                       # default scenario set
    python -m backend.tools.bench_game_load --scenarios idle,card8,card6,gpu50
    python -m backend.tools.bench_game_load --scenario "mine=--card-gb 7 --gpu-busy 0.4"
    python -m backend.tools.bench_game_load --game-first "--card-gb 6"  # small card from boot
    python -m backend.tools.bench_game_load --cpus 0-5 --scenarios idle,cpuall  # a 6-thread CPU

Each scenario runs the engine twice: paced at ``--pace-kps`` keys/s (what the session asks
for, so the game sees a realistic desk) and flat out (the card's capacity). Reported per run:
ms/key p50/p95, keys/s, calls over 100 ms, the "game"'s frame rate alone and beside the
desk, whole-card VRAM (NVML), and the per-process Dedicated / Shared GPU memory Windows
reports (``\\GPU Process Memory(*)``). Shared Usage growing under a squeeze is WDDM placing
or evicting allocations to system RAM -- the crawl a small card hits.

Footprint: after warm-up, a stage-by-stage VRAM breakdown (CUDA context, DiT fp32, SD-VAE,
TinyVAE, graph fp16 copies, graph pool/activations).

Models come from ``VTM_NOBLE_ROOT`` (or ``--root``); in a git worktree without models the
main checkout is used. Results print as a table and are saved as JSON (``--json``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

GIB = 1024**3
MIB = 1024**2
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[2]

SCENARIOS: dict[str, str] = {
    "idle": "",
    # This card acting as an 8 / 6 GB one: the desk and desktop share what is left.
    "card8": "--card-gb 8",
    "card6": "--card-gb 6",
    # A game that fills the card to its budget, then one that asks for more than is there
    # (WDDM has to keep some of someone's allocations in system RAM).
    "full": "--leave-gb 0.25",
    "over1": "--leave-gb -1 --vram-mode force",
    "over3": "--leave-gb -3 --vram-mode force",
    "gpu50": "--gpu-busy 0.5",
    "gpu90": "--gpu-busy 0.9",
    "full+gpu50": "--leave-gb 0.25 --gpu-busy 0.5",
    "over1+gpu50": "--leave-gb -1 --vram-mode force --gpu-busy 0.5",
    "cpuall": f"--cpu-threads {os.cpu_count() or 8} --cpu-busy 1.0",
}
DEFAULT_SET = "idle,card8,card6,full,over1,over3,gpu50,gpu90,full+gpu50,over1+gpu50,cpuall,idle"


# --------------------------------------------------------------------------- setup
def _resolve_root(cli: str) -> Path:
    if cli:
        return Path(cli).resolve()
    env = os.environ.get("VTM_NOBLE_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    if any((PKG_ROOT / "models" / "dit").glob("*.pt")):
        return PKG_ROOT
    try:  # git worktree: models live in the main checkout
        common = subprocess.run(
            ["git", "-C", str(PKG_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        main = Path(common).parent
        if any((main / "models" / "dit").glob("*.pt")):
            return main
    except Exception:
        pass
    return PKG_ROOT


def _mirror_desk_env(alloc_conf: str | None) -> None:
    """Same thread caps / allocator config backend.__main__ sets before torch loads."""
    from backend.__main__ import desk_thread_env

    for key, val in desk_thread_env(os.cpu_count()).items():
        os.environ.setdefault(key, val)
    if alloc_conf is not None:
        if alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    elif os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_PRIORITY = {"idle": 0x40, "below": 0x4000, "normal": 0x20, "above": 0x8000, "high": 0x80}
_GPU_PRIORITY = {"idle": 0, "below": 1, "normal": 2, "above": 3, "high": 4, "realtime": 5}


def set_cpu_priority(pid: int, name: str) -> None:
    import psutil

    psutil.Process(pid).nice(_PRIORITY[name])


def set_gpu_priority(pid: int, name: str) -> int:
    """D3DKMTSetProcessSchedulingPriorityClass: the WDDM GPU scheduler's per-process class.

    Returns the NTSTATUS (0 = ok). HIGH/REALTIME need SeIncreaseBasePriorityPrivilege.
    """
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x0200 | 0x0400, False, pid)  # SET_INFORMATION | QUERY_INFORMATION
    if not h:
        return -1
    try:
        fn = ctypes.WinDLL("gdi32").D3DKMTSetProcessSchedulingPriorityClass
        fn.argtypes = [wintypes.HANDLE, ctypes.c_int]
        fn.restype = ctypes.c_long
        return int(fn(h, _GPU_PRIORITY[name]))
    finally:
        k32.CloseHandle(h)


# --------------------------------------------------------------------------- sampling
class NvmlSampler:
    """Whole-card memory.used and utilization every 0.25 s (pynvml, else nvidia-smi)."""

    def __init__(self, index: int = 0) -> None:
        self.samples: list[tuple[float, int, int]] = []  # (t, used_bytes, util%)
        self._stop = threading.Event()
        self._nv = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            self._nv = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def read(self) -> tuple[int, int]:
        if self._nv is not None:
            mem = self._nv.nvmlDeviceGetMemoryInfo(self._h)
            util = self._nv.nvmlDeviceGetUtilizationRates(self._h)
            return int(mem.used), int(util.gpu)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        used, util = (int(x.strip()) for x in out.split(","))
        return used * MIB, util

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                used, util = self.read()
                self.samples.append((time.perf_counter(), used, util))
            except Exception:
                pass
            self._stop.wait(0.25)

    def window(self, t0: float, t1: float) -> dict[str, Any]:
        rows = [s for s in self.samples if t0 <= s[0] <= t1]
        if not rows:
            return {}
        return {
            "card_used_max_gib": max(r[1] for r in rows) / GIB,
            "gpu_util_avg": sum(r[2] for r in rows) / len(rows),
        }

    def stop(self) -> None:
        self._stop.set()


_COUNTER_RE = re.compile(
    r"gpu (process|adapter) memory\((.*?)\)\\(dedicated|shared) usage\|([0-9.eE+-]+)", re.I
)
_INST_RE = re.compile(r"pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)", re.I)


class WddmSampler:
    """Windows GPU memory counters once a second, via a persistent ``Get-Counter``."""

    def __init__(self) -> None:
        self.ticks: list[tuple[float, dict[str, Any]]] = []
        self._proc: subprocess.Popen[str] | None = None
        self.restart()

    def restart(self) -> None:
        """Get-Counter expands (*) once, at start: restart it to see a process born since."""
        if os.name != "nt":
            return
        self.stop()
        ps = (
            "$ErrorActionPreference='SilentlyContinue';"
            "Get-Counter -Counter '\\GPU Process Memory(*)\\Dedicated Usage',"
            "'\\GPU Process Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Dedicated Usage',"
            "'\\GPU Adapter Memory(*)\\Shared Usage' -SampleInterval 1 -Continuous | "
            "ForEach-Object { foreach ($s in $_.CounterSamples) { if ($s.CookedValue -gt 0) "
            "{ [Console]::Out.WriteLine($s.Path + '|' + $s.CookedValue) } }; "
            "[Console]::Out.WriteLine('TICK'); [Console]::Out.Flush() }"
        )
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read, args=(self._proc,), daemon=True).start()

    def _read(self, proc: subprocess.Popen[str]) -> None:
        cur: dict[str, Any] = {"proc": {}, "adapter": {}}
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line == "TICK":
                self.ticks.append((time.perf_counter(), cur))
                cur = {"proc": {}, "adapter": {}}
                continue
            m = _COUNTER_RE.search(line)
            if not m:
                continue
            kind, inst, which, val = m.group(1).lower(), m.group(2), m.group(3).lower(), float(m.group(4))
            if kind == "process":
                im = _INST_RE.search(inst)
                if not im:
                    continue
                key = (int(im.group(1)), im.group(2).lower())
                cur["proc"].setdefault(key, {})[which] = val
            else:
                luid = inst.lower().split("_phys")[0]
                cur["adapter"].setdefault(luid, {})[which] = val

    def wait_ticks(self, n: int = 1, timeout: float = 8.0) -> None:
        start = len(self.ticks)
        end = time.time() + timeout
        while len(self.ticks) < start + n and time.time() < end:
            time.sleep(0.1)

    def luid_of(self, pid: int) -> str | None:
        for _t, tick in reversed(self.ticks[-5:]):
            best = None
            for (p, luid), v in tick["proc"].items():
                if p == pid and v.get("dedicated", 0) > (best[1] if best else 0):
                    best = (luid, v.get("dedicated", 0))
            if best:
                return best[0]
        return None

    def proc(self, pid: int, luid: str | None, t0: float | None = None, t1: float | None = None) -> dict[str, float]:
        """Max dedicated/shared GiB of ``pid`` on the adapter over [t0, t1] (latest tick if None)."""
        ticks = self.ticks[-1:] if t0 is None else [x for x in self.ticks if t0 <= x[0] <= (t1 or 1e18)]
        ded = sh = 0.0
        for _t, tick in ticks:
            for (p, lu), v in tick["proc"].items():
                if p == pid and (luid is None or lu == luid):
                    ded = max(ded, v.get("dedicated", 0.0))
                    sh = max(sh, v.get("shared", 0.0))
        return {"dedicated_gib": ded / GIB, "shared_gib": sh / GIB}

    def adapter(self, luid: str | None, t0: float, t1: float) -> dict[str, float]:
        ded = sh = 0.0
        for _t, tick in (x for x in self.ticks if t0 <= x[0] <= t1):
            for lu, v in tick["adapter"].items():
                if luid is None or lu == luid:
                    ded = max(ded, v.get("dedicated", 0.0))
                    sh = max(sh, v.get("shared", 0.0))
        return {"adapter_dedicated_gib": ded / GIB, "adapter_shared_gib": sh / GIB}

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


# --------------------------------------------------------------------------- stress child
class Stress:
    """One stress_game child; parses its STRESS lines."""

    def __init__(self, extra: str, *, cpus: str = "", priority: str = "", gpu_priority: str = "") -> None:
        fd, self.stop_file = tempfile.mkstemp(prefix="vtm_stress_", suffix=".stop")
        os.close(fd)
        os.unlink(self.stop_file)
        args = [sys.executable, "-m", "backend.tools.stress_game", "--stop-file", self.stop_file,
                "--parent-pid", str(os.getpid()), *shlex.split(extra)]
        if cpus and "--cpus" not in extra:
            args += ["--cpus", cpus]
        self.events: list[tuple[float, dict[str, Any]]] = []
        self.log: list[str] = []
        self.proc = subprocess.Popen(
            args, cwd=str(PKG_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        # A venv's python.exe is a launcher; the real interpreter is its child and gives
        # its own pid in the ready line.
        self.pid = self.proc.pid
        self._priority, self._gpu_priority = priority, gpu_priority
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip()
            if line.startswith("STRESS "):
                try:
                    self.events.append((time.perf_counter(), json.loads(line[7:])))
                except ValueError:
                    pass
            else:
                self.log.append(line)

    def event(self, name: str) -> dict[str, Any] | None:
        for _t, ev in self.events:
            if ev.get("event") == name:
                return ev
        return None

    def wait_ready(self, timeout: float = 180.0) -> dict[str, Any]:
        end = time.time() + timeout
        while time.time() < end:
            ev = self.event("ready")
            if ev is not None:
                self.pid = int(ev.get("pid") or self.pid)
                if self._priority:
                    set_cpu_priority(self.pid, self._priority)
                if self._gpu_priority:
                    st = set_gpu_priority(self.pid, self._gpu_priority)
                    print(f"[bench] game GPU class {self._gpu_priority}: status {st:#x}", flush=True)
                return ev
            if self.proc.poll() is not None:
                raise RuntimeError("stress_game exited early:\n" + "\n".join(self.log[-20:]))
            time.sleep(0.1)
        raise TimeoutError("stress_game did not get ready")

    def game_stats(self, t0: float, t1: float) -> dict[str, float]:
        rows = [ev for t, ev in self.events if ev.get("event") == "stats" and t0 < t <= t1]
        if not rows:
            return {}
        fps = sum(r["fps"] for r in rows) / len(rows)
        return {
            "game_fps": fps,
            "game_work_ms_p50": sorted(r["work_ms_p50"] for r in rows)[len(rows) // 2],
            "game_work_ms_p95": max(r["work_ms_p95"] for r in rows),
        }

    def stop(self, timeout: float = 15.0) -> None:
        if self.proc.poll() is None:
            Path(self.stop_file).touch()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        try:
            os.unlink(self.stop_file)
        except OSError:
            pass


# --------------------------------------------------------------------------- engine
def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else 0.0


def _bytes_of(module: Any) -> int:
    if module is None:
        return 0
    mod = getattr(module, "_orig_mod", module)
    total = 0
    seen: set[int] = set()
    for t in list(mod.parameters()) + list(mod.buffers()):
        if t.device.type == "cuda" and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


class Bench:
    def __init__(self, args: argparse.Namespace, nvml: NvmlSampler, wddm: WddmSampler) -> None:
        self.args, self.nvml, self.wddm = args, nvml, wddm
        self.luid: str | None = None
        self.footprint: list[dict[str, Any]] = []
        self.engine: Any = None
        self.kps: Any = None
        self.batch = int(args.batch)
        self.pre_game: Stress | None = None

    def snap(self, label: str, *, settle: float = 0.0) -> dict[str, Any]:
        import torch

        if settle:
            time.sleep(settle)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.wddm.wait_ticks(1)
        used, _util = self.nvml.read()
        row = {
            "stage": label,
            "allocated_gib": torch.cuda.memory_allocated() / GIB if torch.cuda.is_initialized() else 0.0,
            "reserved_gib": torch.cuda.memory_reserved() / GIB if torch.cuda.is_initialized() else 0.0,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_initialized() else 0.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB if torch.cuda.is_initialized() else 0.0,
            "card_used_gib": used / GIB,
            **self.wddm.proc(os.getpid(), self.luid),
        }
        self.footprint.append(row)
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        print(f"[bench] {label:<18} alloc {row['allocated_gib']:.2f}  reserved {row['reserved_gib']:.2f}  "
              f"peak {row['peak_allocated_gib']:.2f}/{row['peak_reserved_gib']:.2f}  "
              f"card {row['card_used_gib']:.2f}  proc ded {row['dedicated_gib']:.2f} / shared "
              f"{row['shared_gib']:.2f} GiB", flush=True)
        return row

    def setup(self) -> dict[str, Any]:
        import numpy as np
        import torch

        a = self.args
        self.snap("before CUDA")
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        self.wddm.wait_ticks(2)
        self.luid = self.wddm.luid_of(os.getpid())
        self.snap("CUDA context")

        from backend import engine as eng
        from backend.paths import models_dir, refs_dir

        ckpt = Path(a.checkpoint) if a.checkpoint else models_dir() / "VTM-1.5.1.pt"
        if not ckpt.is_file():
            ckpt = models_dir() / ckpt.name
        e = eng.StreamEngine(checkpoint=ckpt, fast_mode=True, compile_model=a.compile)
        self.engine = e
        t0 = time.perf_counter()

        def on_load(key: str, _label: str) -> None:
            if key == "vae":
                self.snap("+ DiT fp32")

        e.load(on_stage=on_load)
        self.snap("+ SD-VAE")
        ref = Path(a.ref) if a.ref else refs_dir() / "upload.png"
        kps_path = ref.with_name(ref.stem + "_keypoints.npy")
        e.set_reference(ref, kps_path if kps_path.is_file() else None)
        self.snap("+ reference")
        e.set_hold_last(bool(a.hold_last))
        e.num_steps = 1
        e.set_stream_batch_size(self.batch)

        def on_warm(key: str, _label: str) -> None:
            if key == "graph":
                self.snap("+ TinyVAE")

        warm_err = ""
        try:
            e.warmup(num_steps=1, batch_size=self.batch, on_stage=on_warm)
        except Exception as exc:  # the stream steps down to batch 1 on OOM
            warm_err = str(exc).splitlines()[0][:200]
            print(f"[bench] warmup at batch {self.batch} failed: {warm_err}; retrying at 1", flush=True)
            self.batch = 1
            e.set_stream_batch_size(1)
            e._restore_eager_model()
            e._compile_failed = False
            torch.cuda.empty_cache()
            e.warmup(num_steps=1, batch_size=1, on_stage=on_warm)
        warm_s = time.perf_counter() - t0
        self.snap("+ graph (warm)", settle=0.5)

        base = np.asarray(e._ref_keypoints, dtype=np.float32)
        rng = np.random.default_rng(0)
        self.kps = [
            np.stack([base] * self.batch, 0) if self.batch > 1 else base.copy() for _ in range(16)
        ]
        for k in self.kps:  # small per-call jitter, as a live face gives
            k[..., :2] += rng.normal(0, 0.004, size=k[..., :2].shape).astype(np.float32)
        for _ in range(5):
            self.call(0)

        gf = e._graph_frames.get(e.graph_mode) if e._graph_frames else None
        info = {
            "checkpoint": ckpt.name,
            "batch": self.batch,
            "warmup_error": warm_err,
            "warm_s": warm_s,
            "speed_mode": e.active_speed_mode,
            "graph_dtype": getattr(gf, "dtype_name", None),
            "graph_decoder": getattr(gf, "decoder_name", None),
            "graph_compiled": bool(getattr(gf, "compiled", False)),
            "decode_backend": e.last_decode_backend,
            "weights_gib": {
                "dit_fp32": _bytes_of(e._eager_model or e.model) / GIB,
                "sd_vae": _bytes_of(e.vae) / GIB,
                "sd_vae_encoder": _bytes_of(getattr(e.vae, "encoder", None)) / GIB,
                "sd_vae_decoder": _bytes_of(getattr(e.vae, "decoder", None)) / GIB,
                "tiny_vae": _bytes_of(e.vae_tiny) / GIB,
                "graph_dit_copy": _bytes_of(getattr(gf, "dit", None)) / GIB,
                "graph_decoder_copy": _bytes_of(getattr(gf, "dec", None)) / GIB,
            },
            "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        }
        return info

    def call(self, i: int) -> float:
        t0 = time.perf_counter()
        self.engine.generate_batch_from_keypoints(
            self.kps[i % len(self.kps)], num_steps=1, sanitize="constrained"
        )
        return time.perf_counter() - t0

    def run(self, seconds: float, pace_kps: float) -> dict[str, Any]:
        """Back-to-back (pace 0) or paced calls for ``seconds``."""
        times: list[float] = []
        interval = self.batch / pace_kps if pace_kps > 0 else 0.0
        t_begin = time.perf_counter()
        next_at = t_begin
        i = 0
        while True:
            now = time.perf_counter()
            if now - t_begin >= seconds:
                break
            if interval and now < next_at:
                time.sleep(next_at - now)
            start = time.perf_counter()
            times.append(self.call(i))
            next_at = max(start + interval, time.perf_counter()) if interval else 0.0
            i += 1
        t_end = time.perf_counter()
        per_key = [t / self.batch * 1000.0 for t in times]
        return {
            "t0": t_begin,
            "t1": t_end,
            "calls": len(times),
            "keys_per_s": len(times) * self.batch / (t_end - t_begin),
            "ms_per_key_p50": _pct(per_key, 0.5),
            "ms_per_key_p95": _pct(per_key, 0.95),
            "call_ms_max": max(times) * 1000.0 if times else 0.0,
            "calls_over_100ms": sum(1 for t in times if t > 0.1),
        }


def run_scenario(bench: Bench, name: str, extra: str, args: argparse.Namespace) -> dict[str, Any]:
    print(f"\n[bench] === {name}: {extra or '(no load)'}", flush=True)
    res: dict[str, Any] = {"scenario": name, "stress_args": extra}
    stress = None
    if extra and args.idle_ref > 0:
        # Unloaded reference right before this scenario: another job on the GPU moves the
        # baseline, and the ratio to this is what the load itself costs.
        r = bench.run(args.idle_ref, 0.0)
        res["idle_ref"] = {k: r[k] for k in ("keys_per_s", "ms_per_key_p50", "ms_per_key_p95")}
        res["idle_ref"].update(bench.nvml.window(r["t0"], r["t1"]))
    try:
        if extra:
            stress = Stress(extra, cpus=args.cpus, priority=args.game_priority,
                            gpu_priority=args.game_gpu_priority)
            ready = stress.wait_ready()
            res["stress_free_before_gib"] = ready.get("free_before", 0) / GIB
            res["stress_held_gib"] = ready.get("held_bytes", 0) / GIB
            res["stress_target_gib"] = ready.get("target_bytes", 0) / GIB
            res["stress_reached"] = ready.get("reached", True)
            res["stress_stop"] = ready.get("stop_reason", "")
            bench.wddm.restart()
            bench.wddm.wait_ticks(1, timeout=15)
            time.sleep(args.settle)
            if "--gpu-busy" in extra:
                t0 = time.perf_counter()
                time.sleep(3.2)
                res["game_alone"] = stress.game_stats(t0, time.perf_counter())
        for label, pace, secs in (("paced", args.pace_kps, args.paced_s), ("max", 0.0, args.max_s)):
            if secs <= 0:
                continue
            r = bench.run(secs, pace)
            t0, t1 = r.pop("t0"), r.pop("t1")
            r.update(bench.nvml.window(t0, t1))
            r.update({f"desk_{k}": v for k, v in bench.wddm.proc(os.getpid(), bench.luid, t0, t1).items()})
            r.update(bench.wddm.adapter(bench.luid, t0, t1))
            game = stress or bench.pre_game
            if game is not None:
                r.update({f"game_{k}": v for k, v in bench.wddm.proc(game.pid, bench.luid, t0, t1).items()})
                r.update(game.game_stats(t0, t1))
            res[label] = r
            print(f"[bench] {name:<16} {label:<5} {r['ms_per_key_p50']:6.2f} ms/key p50 "
                  f"(p95 {r['ms_per_key_p95']:6.2f}) {r['keys_per_s']:6.1f} keys/s  "
                  f"desk shared {r.get('desk_shared_gib', 0):.2f} GiB  "
                  f"game fps {r.get('game_fps', float('nan')):.1f}", flush=True)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
        print(f"[bench] {name} failed: {res['error']}", flush=True)
    finally:
        if stress is not None:
            stress.stop()
            time.sleep(2.0)
    return res


def print_table(results: list[dict[str, Any]], batch: int) -> None:
    def f(v: Any, w: int, fmt: str = ".1f") -> str:
        return f"{format(v, fmt) if isinstance(v, (int, float)) else '-':>{w}}"

    hdr = (f"{'scenario':<14}{'held':>6}{'ref':>6} |{'pace k/s':>9}{'ms/key':>7}{'p95':>7} |"
           f"{'max k/s':>8}{'ms/key':>7}{'p95':>7}{'>100':>5} |{'desk sh':>8}{'card sh':>8} |"
           f"{'game':>6}{'+pace':>6}{'+max':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        p, m = r.get("paced", {}), r.get("max", {})
        print(
            f"{r['scenario']:<14}{f(r.get('stress_held_gib'), 6)}{f(r.get('idle_ref', {}).get('ms_per_key_p50'), 6)} |"
            f"{f(p.get('keys_per_s'), 9)}{f(p.get('ms_per_key_p50'), 7)}{f(p.get('ms_per_key_p95'), 7)} |"
            f"{f(m.get('keys_per_s'), 8)}{f(m.get('ms_per_key_p50'), 7)}{f(m.get('ms_per_key_p95'), 7)}"
            f"{f(p.get('calls_over_100ms', 0) + m.get('calls_over_100ms', 0), 5, 'd')} |"
            f"{f(max(p.get('desk_shared_gib', 0), m.get('desk_shared_gib', 0)), 8, '.2f')}"
            f"{f(max(p.get('adapter_shared_gib', 0), m.get('adapter_shared_gib', 0)), 8, '.2f')} |"
            f"{f(r.get('game_alone', {}).get('game_fps'), 6)}{f(p.get('game_fps'), 6)}{f(m.get('game_fps'), 6)}"
            + (f"   ERROR {r['error']}" if r.get("error") else "")
        )
    print(f"(batch {batch}. held = GiB the game holds; ref = unloaded ms/key just before; pace = "
          "session key rate, max = back-to-back; >100 = calls over 100 ms; desk/card sh = shared "
          "(system RAM) GPU memory, GiB; game = its fps alone / beside the paced / max desk)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="", help="install root with models/ (default: auto)")
    ap.add_argument("--checkpoint", default="", help="models/dit file (default VTM-1.5.1.pt)")
    ap.add_argument("--ref", default="", help="reference still (default models/refs/upload.png)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--compile", type=int, default=1, help="torch.compile the graph (session default on)")
    ap.add_argument("--hold-last", type=int, default=1)
    ap.add_argument("--scenarios", default=DEFAULT_SET, help=f"comma list from: {', '.join(SCENARIOS)}")
    ap.add_argument("--scenario", action="append", default=[], help='custom "name=stress_game args"')
    ap.add_argument("--game-first", default="", help="stress_game args held from before the engine loads")
    ap.add_argument("--pace-kps", type=float, default=40.0, help="paced run keys/s (session Max FPS)")
    ap.add_argument("--paced-s", type=float, default=10.0)
    ap.add_argument("--max-s", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--idle-ref", type=float, default=4.0, help="seconds of unloaded max run before each load (0 = off)")
    ap.add_argument("--cpus", default="", help="pin bench + stress to these CPUs, e.g. 0-5")
    ap.add_argument("--desk-priority", choices=list(_PRIORITY), default="")
    ap.add_argument("--desk-gpu-priority", choices=list(_GPU_PRIORITY), default="")
    ap.add_argument("--game-priority", choices=list(_PRIORITY), default="")
    ap.add_argument("--game-gpu-priority", choices=list(_GPU_PRIORITY), default="")
    ap.add_argument("--alloc-conf", default=None, help="PYTORCH_CUDA_ALLOC_CONF ('' = unset)")
    ap.add_argument("--json", default="", help="write results here")
    args = ap.parse_args(argv)

    root = _resolve_root(args.root)
    os.environ["VTM_NOBLE_ROOT"] = str(root)
    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
    _mirror_desk_env(args.alloc_conf)
    from backend.paths import configure_torch_compile_cache

    configure_torch_compile_cache()  # before torch loads, as backend.engine does
    if args.cpus:
        from backend.tools.stress_game import parse_cpus, pin_cpus

        pin_cpus(parse_cpus(args.cpus))
    if args.desk_priority:
        set_cpu_priority(os.getpid(), args.desk_priority)
    if args.desk_gpu_priority:
        st = set_gpu_priority(os.getpid(), args.desk_gpu_priority)
        print(f"[bench] desk GPU scheduling class {args.desk_gpu_priority}: status {st:#x}", flush=True)

    plan: list[tuple[str, str]] = []
    for name in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        if name not in SCENARIOS:
            ap.error(f"unknown scenario {name!r}")
        plan.append((name, SCENARIOS[name]))
    for spec in args.scenario:
        name, _, extra = spec.partition("=")
        plan.append((name.strip(), extra.strip()))

    print(f"[bench] root {root}  pid {os.getpid()}  PYTORCH_CUDA_ALLOC_CONF="
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}", flush=True)
    nvml = NvmlSampler()
    wddm = WddmSampler()
    wddm.wait_ticks(1, timeout=15)
    pre_game = None
    out: dict[str, Any] = {"root": str(root), "args": vars(args), "results": []}
    # Other GPU work on the PC (OBS, a browser, another job) skews every number; say how much.
    t_bg = time.perf_counter()
    time.sleep(3.0)
    out["background"] = nvml.window(t_bg, time.perf_counter())
    print(f"[bench] background before load: {out['background']}", flush=True)
    try:
        if args.game_first:
            pre_game = Stress(args.game_first, cpus=args.cpus, priority=args.game_priority,
                              gpu_priority=args.game_gpu_priority)
            ready = pre_game.wait_ready()
            out["game_first"] = {k: ready.get(k) for k in ("held_bytes", "target_bytes", "reached", "stop_reason")}
            print(f"[bench] game-first holds {ready.get('held_bytes', 0) / GIB:.2f} GiB", flush=True)
            wddm.restart()
            wddm.wait_ticks(1, timeout=15)
        bench = Bench(args, nvml, wddm)
        bench.pre_game = pre_game
        import torch

        out["gpu"] = torch.cuda.get_device_name(0)
        out["setup"] = bench.setup()
        out["footprint"] = bench.footprint
        for name, extra in plan:
            out["results"].append(run_scenario(bench, name, extra, args))
        print_table(out["results"], bench.batch)
    finally:
        if pre_game is not None:
            pre_game.stop()
        wddm.stop()
        nvml.stop()
        path = Path(args.json) if args.json else Path(tempfile.gettempdir()) / "vtm_bench" / (
            time.strftime("bench_game_load-%Y%m%d-%H%M%S.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"[bench] results -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
