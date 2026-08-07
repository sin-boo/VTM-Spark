"""Download/cache the fast encoder stack (SD-VAE + CLIP).

Also optionally warms the quality stack (FLUX.2 VAE + T5Gemma) if already used.

Usage:
  cd /d D:\\ai_vtuber\\pipeline\\i1\\torch_train
  .venv\\Scripts\\python scripts\\download_fast_stack.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache SD-VAE + CLIP (and optionally quality stack).")
    parser.add_argument(
        "--also-quality",
        action="store_true",
        help="Also ensure FLUX.2 VAE + T5Gemma are cached.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from text_encoder.text_encoder import TextEncoder
    from vae.vae import load_vae

    class Cfg:
        def __init__(self, vae_type: str):
            self.vae_type = vae_type

    device = torch.device("cpu")
    print("Downloading / caching fast stack: stabilityai/sd-vae-ft-mse + openai/clip-vit-large-patch14")
    _ = load_vae(Cfg("sd"), device, dtype=torch.float32)
    _ = TextEncoder(None, "CLIP", None, weight_dtype=torch.float32, device=device)
    print("Fast stack cached.")

    if args.also_quality:
        print("Downloading / caching quality stack: FLUX.2 VAE + T5Gemma")
        _ = load_vae(Cfg("flux2"), device, dtype=torch.float32)
        _ = TextEncoder(None, "T5Gemma", None, weight_dtype=torch.float32, device=device)
        print("Quality stack cached.")

    print("Done.")


if __name__ == "__main__":
    main()
