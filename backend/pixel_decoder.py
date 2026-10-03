"""Fast TAESD-style latent decoder with a PixelShuffle head.

Same input as the Hybrid TinyVAE the app uses (SD latents already * 0.18215, 4x96x96) and the
same output range ([-1, 1] RGB at 768x768), but the last stage stops at 384^2 and emits
3*2*2 channels that PixelShuffle turns into 768^2 -- no convolutions at full resolution.

Stages 96^2/192^2 (and the first two 384^2 blocks) have the TinyVAE decoder's exact layout,
so they are warm-started from its weights (see `from_tiny_vae`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
        )
        self.skip = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


class PixelShuffleDecoder(nn.Module):
    """chans/blocks per stage starting at the latent resolution; the last stage feeds the shuffle head."""

    def __init__(self, chans=(64, 64, 64), blocks=(3, 3, 2), shuffle: int = 2):
        super().__init__()
        self.arch = {"chans": list(chans), "blocks": list(blocks), "shuffle": int(shuffle)}
        layers: list[nn.Module] = [nn.Conv2d(4, chans[0], 3, padding=1), nn.ReLU()]
        c = chans[0]
        for i, (ch, nb) in enumerate(zip(chans, blocks)):
            for _ in range(nb):
                layers.append(Block(c, ch))
                c = ch
            if i < len(chans) - 1:
                layers += [nn.Upsample(scale_factor=2), nn.Conv2d(c, chans[i + 1], 3, padding=1, bias=False)]
                c = chans[i + 1]
        layers.append(nn.Conv2d(c, 3 * shuffle * shuffle, 3, padding=1))
        layers.append(nn.PixelShuffle(shuffle))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: scaled SD latents (B,4,h,w) -> RGB in ~[-1, 1] (B,3,8h,8w)."""
        return self.layers(torch.tanh(z / 3) * 3)

    @classmethod
    def from_tiny_vae(cls, tiny_decoder: nn.Module, **kw) -> "PixelShuffleDecoder":
        """Copy every TinyVAE decoder weight whose name and shape match (stages before the head)."""
        m = cls(**kw)
        src = tiny_decoder.state_dict()
        dst = m.state_dict()
        copied = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
        dst.update(copied)
        m.load_state_dict(dst)
        m.copied_keys = sorted(copied)
        return m

    @classmethod
    def load(cls, path, map_location="cpu") -> "PixelShuffleDecoder":
        ck = torch.load(path, map_location=map_location, weights_only=True)
        m = cls(**ck["arch"])
        m.load_state_dict(ck["state_dict"])
        return m

    def save(self, path, **extra) -> None:
        torch.save({"arch": self.arch, "state_dict": self.state_dict(), **extra}, path)
