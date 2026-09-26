"""One-shot load / warmup / generate timing. Not part of the test suite."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from backend.character_pack import read_character_pack
from backend.engine import StreamEngine


CKPT = ROOT / "models" / "dit" / "VTM-1.5.1.pt"
CHAR = ROOT / "characters" / "ChatGPT-Image-Aug-3-2026-11_36_05-PM.vtm"
GEN_RUNS = 8


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _mem() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free_b, total_b = torch.cuda.mem_get_info()
    used = (total_b - free_b) / (1024**3)
    alloc = torch.cuda.memory_allocated() / (1024**3)
    return f"GPU used {used:.2f} GiB (alloc {alloc:.2f} / {total_b / (1024**3):.2f})"


def _sec(t0: float) -> float:
    return time.perf_counter() - t0


def bench_generate(engine: StreamEngine, batch: int, runs: int = GEN_RUNS) -> dict:
    kps = np.asarray(engine._ref_keypoints, dtype=np.float32)
    if batch > 1:
        kps = np.stack([kps] * batch, axis=0)
    times: list[float] = []
    for i in range(runs):
        _images, elapsed = engine.generate_batch_from_keypoints(
            kps, num_steps=2, sanitize="none"
        )
        times.append(float(elapsed))
        print(f"  gen[{batch}] {i + 1}/{runs}: {elapsed:.3f}s  ({batch / elapsed:.2f} img/s)", flush=True)
    steady = times[2:] if len(times) > 3 else times[1:]
    mean = sum(steady) / max(len(steady), 1)
    return {
        "batch": batch,
        "runs": times,
        "steady_s": mean,
        "dit_hz": (1.0 / mean) if mean > 0 else 0.0,
        "img_hz": (batch / mean) if mean > 0 else 0.0,
    }


def main() -> int:
    if not CKPT.is_file():
        print(f"missing checkpoint: {CKPT}")
        return 1
    if not CHAR.is_file():
        print(f"missing character: {CHAR}")
        return 1
    if not torch.cuda.is_available():
        print("CUDA is required for this bench")
        return 1

    print(f"[{_now()}] device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[{_now()}] ckpt={CKPT.name}  compile=on  fast=on", flush=True)
    print(f"[{_now()}] {_mem()}", flush=True)

    engine = StreamEngine(
        checkpoint=CKPT,
        device="cuda",
        fast_mode=True,
        compile_model=True,
    )

    print(f"\n[{_now()}] LOAD DiT + SD-VAE …", flush=True)
    t0 = time.perf_counter()
    engine.load()
    t_load = _sec(t0)
    print(f"[{_now()}] load {t_load:.2f}s  {_mem()}", flush=True)

    print(f"\n[{_now()}] LOAD character latents …", flush=True)
    pack = read_character_pack(CHAR)
    t0 = time.perf_counter()
    engine.load_encoded_reference(
        keypoints=pack.keypoints,
        ref_latent=pack.ref_latent,
        ref_face_latent=pack.ref_face_latent,
        skip_crop=pack.skip_crop,
        path=CHAR,
    )
    t_ref = _sec(t0)
    print(f"[{_now()}] character {t_ref:.2f}s  {_mem()}", flush=True)

    # Old first-start: compile at batch=1, drop wrapper, compile at batch=2.
    print(f"\n[{_now()}] OLD warmup compile batch=1 …", flush=True)
    t0 = time.perf_counter()
    engine.warmup(num_steps=2, batch_size=1)
    t_old_b1 = _sec(t0)
    print(f"[{_now()}] old B=1 {t_old_b1:.2f}s  {_mem()}", flush=True)

    print(f"\n[{_now()}] OLD drop compiled graph, compile batch=2 …", flush=True)
    engine._restore_eager_model()
    engine._compile_failed = False
    engine._compile_verified = False
    t0 = time.perf_counter()
    engine.warmup(num_steps=2, batch_size=2)
    t_old_b2 = _sec(t0)
    print(f"[{_now()}] old B=2 {t_old_b2:.2f}s  {_mem()}", flush=True)

    t_old = t_old_b1 + t_old_b2
    t_new = t_old_b2  # new path compiles once at the batch we actually stream

    print(f"\n[{_now()}] GENERATE batch×2 (new/current streaming path) …", flush=True)
    gen2 = bench_generate(engine, 2)

    print(f"\n[{_now()}] GENERATE batch×1 (second captured shape, no full recompile) …", flush=True)
    t0 = time.perf_counter()
    engine.warmup(num_steps=2, batch_size=1)
    t_b1_recapture = _sec(t0)
    print(f"[{_now()}] B=1 recapture {t_b1_recapture:.2f}s  {_mem()}", flush=True)
    gen1 = bench_generate(engine, 1)

    print("\n========== RESULTS ==========")
    print(f"GPU              {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint       {CKPT.name}")
    print(f"Load DiT+VAE     {t_load:.2f}s")
    print(f"Load character   {t_ref:.2f}s")
    print(f"OLD warmup       {t_old:.2f}s  (B=1 compile {t_old_b1:.2f}s + drop + B=2 compile {t_old_b2:.2f}s)")
    print(f"NEW warmup       {t_new:.2f}s  (compile once at B=2)")
    saved = t_old - t_new
    if t_old > 0:
        print(f"Warmup saved     {saved:.2f}s  ({100.0 * saved / t_old:.0f}% faster first compile)")
    print(f"B=1 recapture    {t_b1_recapture:.2f}s  (keep wrapper, second graph)")
    print(
        f"Gen batch×2      {gen2['dit_hz']:.2f} DiT/s  |  {gen2['img_hz']:.2f} img/s  "
        f"(steady {gen2['steady_s']:.3f}s / call)"
    )
    print(
        f"Gen batch×1      {gen1['dit_hz']:.2f} DiT/s  |  {gen1['img_hz']:.2f} img/s  "
        f"(steady {gen1['steady_s']:.3f}s / call)"
    )
    print(f"VRAM after       {_mem()}")
    print("=============================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
