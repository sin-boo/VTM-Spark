"""Whole-frame CUDA graph for the 1-step stream path.

The eager path launches ~2,500 small kernels per frame (pose map, DiT, decode) and syncs
the host ~80 times; on this class of PC the CPU cost of launching them is several times the
GPU work. Here one frame -- pose map, keypoint deltas, DiT step, decode, uint8 -- is captured
once per batch size and replayed with a single launch. Same math as ``denoise_keypoint`` +
``decode_tiny_vae`` for ``num_steps=1`` without CFG, including the hold-last start:
``x = (1-t0)*noise + t0*last``, one Euler step of ``1-t0`` (fresh frames use ``t0=0``).

The DiT and decoder run in fp16 (outputs match the fp32 path to ~57 dB PSNR), or fp32 on
GPUs without fast fp16. Weights are private copies, so the fp32 model the rest of the engine
uses is untouched (and waits in RAM while keys run here). Each replay also reports whether
the frame came out finite (fp16 overflow shows up as NaN/Inf), read back with the image so
it costs no extra sync.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.keypoints import (
    BODY_BONE_INDICES,
    NUM_KEYPOINTS,
    NUM_POSE_CHANNELS,
    POSE_CHANNEL_GROUPS,
    group_indices,
)


class PoseMap(nn.Module):
    """Sync-free, vectorized ``utils.pose_map.rasterize_pose_maps`` (max diff ~3e-4)."""

    def __init__(self, size: int, sigma: float = 1.5, thickness: float = 1.0) -> None:
        super().__init__()
        self.size = int(size)
        self.inv = 1.0 / (2.0 * sigma * sigma + 1e-8)
        self.inv_line = 1.0 / (2.0 * thickness * thickness + 1e-8)
        member = torch.zeros(NUM_KEYPOINTS, NUM_POSE_CHANNELS)
        self.skel_ch = POSE_CHANNEL_GROUPS.index("skeleton")
        self.hair_start = min(i for i, g in enumerate(POSE_CHANNEL_GROUPS) if g.startswith("hair_"))
        for ch, group in enumerate(POSE_CHANNEL_GROUPS):
            if group == "skeleton" or group.startswith("hair_"):
                continue
            for i in group_indices(group):
                member[i, ch] = 1.0
        bones = torch.tensor(BODY_BONE_INDICES, dtype=torch.long)
        grid = torch.arange(self.size, dtype=torch.float32)
        self.register_buffer("member", member[:, : self.hair_start].contiguous())
        self.register_buffer("bone_a", bones[:, 0].clone())
        self.register_buffer("bone_b", bones[:, 1].clone())
        self.register_buffer("yy", grid.view(1, 1, -1, 1))
        self.register_buffer("xx", grid.view(1, 1, 1, -1))

    def forward(self, kps: torch.Tensor, hair: torch.Tensor) -> torch.Tensor:
        n = self.size - 1
        xs = ((kps[..., 0] + 1.0) * 0.5 * n)[..., None, None]
        ys = ((kps[..., 1] + 1.0) * 0.5 * n)[..., None, None]
        vis = kps[..., 3]
        wts = kps[..., 2].clamp(min=0.0) * (vis > 0.5).float()
        g = torch.exp(-((self.xx - xs) ** 2 + (self.yy - ys) ** 2) * self.inv) * wts[..., None, None]
        out = torch.einsum("bkhw,kc->bchw", g, self.member)
        xa, ya, xb, yb = xs[:, self.bone_a], ys[:, self.bone_a], xs[:, self.bone_b], ys[:, self.bone_b]
        both = ((vis[:, self.bone_a] > 0.5) & (vis[:, self.bone_b] > 0.5)).float()[..., None, None]
        dx, dy = xb - xa, yb - ya
        len2 = (dx * dx + dy * dy).clamp(min=1e-4)
        t = (((self.xx - xa) * dx + (self.yy - ya) * dy) / len2).clamp(0.0, 1.0)
        line = torch.exp(-((self.xx - (xa + t * dx)) ** 2 + (self.yy - (ya + t * dy)) ** 2) * self.inv_line)
        skel = (line * both).sum(1, keepdim=True)
        out = torch.cat([out[:, : self.skel_ch], skel, out[:, self.skel_ch + 1 :], hair.clamp(0.0, 1.0)], 1)
        return out.clamp(0.0, 1.0)


def keypoint_deltas_t(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Torch ``utils.keypoints.keypoint_deltas`` over a batch: (dx, dy, score_t, both_visible)."""
    both = (target[..., 3] > 0.5) & (ref[..., 3] > 0.5)
    d = torch.where(both[..., None], target[..., :2] - ref[..., :2], torch.zeros_like(target[..., :2]))
    return torch.cat([d, target[..., 2:3], both[..., None].to(target.dtype)], -1)


class TinyVaeDecode(nn.Module):
    """AutoencoderTiny decode as a module: scaled SD latents -> RGB in [-1, 1]."""

    def __init__(self, vae_tiny: nn.Module) -> None:
        super().__init__()
        self.decoder = vae_tiny.decoder
        cfg = getattr(vae_tiny, "config", None)
        self.scale = float(getattr(cfg, "scaling_factor", 1.0) or 1.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if abs(self.scale - 1.0) > 1e-6:
            z = z / self.scale
        return self.decoder(z)


def _compile(module: nn.Module, mode: str | None) -> nn.Module:
    if not mode:
        return module
    # The graph is captured here; inductor must not add CUDA graphs of its own.
    if "cudagraph" in mode or mode == "reduce-overhead":
        mode = "max-autotune-no-cudagraphs"
    return torch.compile(module, mode=mode, fullgraph=False, dynamic=False)


class GraphedFrame:
    """Replays captured frames. One graph per batch size, sharing a memory pool."""

    def __init__(
        self,
        model: nn.Module,
        decoder: nn.Module,
        *,
        seed: int,
        dtype: torch.dtype = torch.float16,
        compile_mode: str | None = None,
        pose_sigma: float = 1.5,
        device: torch.device | str | None = None,
    ) -> None:
        # ``device`` lets the source model sit in RAM: the copy goes straight to the GPU
        # at graph precision, never as a second fp32 DiT in VRAM.
        device = torch.device(device) if device is not None else next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("GraphedFrame needs CUDA")
        self.device, self.dtype = device, dtype
        dit = copy.deepcopy(getattr(model, "_orig_mod", model))
        dit = dit.to(device=device, dtype=dtype).eval().requires_grad_(False)
        dec = copy.deepcopy(decoder).to(device=device, dtype=dtype).eval().requires_grad_(False)
        dec = dec.to(memory_format=torch.channels_last)
        self.input_size = int(dit.input_size)
        self.in_channels = int(dit.in_channels)
        self.pose_channels = int(getattr(dit, "num_pose_channels", NUM_POSE_CHANNELS))
        self.use_face = bool(getattr(dit, "use_ref_face_tokens", False))
        self.face_size = int(getattr(dit, "ref_face_size", 32) or 32)
        self.dit = _compile(dit, compile_mode)
        self.dec = _compile(dec, compile_mode)
        self.compiled = bool(compile_mode)
        self.pose = PoseMap(self.input_size, sigma=pose_sigma).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        # Same draw as denoise_keypoint(shared_noise=True) with an fp32 model, then cast.
        self.noise1 = torch.randn(
            (1, self.in_channels, self.input_size, self.input_size), generator=gen, device=device, dtype=torch.float32
        ).to(dtype)
        self.pool = torch.cuda.graph_pool_handle()
        self.slots: dict[int, dict[str, Any]] = {}
        self.done = torch.cuda.Event(blocking=True)  # host waits by yielding, not spinning

    # ------------------------------------------------------------------ capture
    def _body(self, s: dict[str, Any]) -> None:
        dt = self.dtype
        t0 = s["t0"]
        w = t0[:, None, None, None].to(dt)
        # Fresh frames (t0=0) never read ``last``: a stale NaN there would survive 0 * NaN.
        x = torch.where(w > 0, (1.0 - w) * s["noise"] + w * s["last"], s["noise"])
        deltas = keypoint_deltas_t(s["kps_t"], s["kps_r"]).to(dt)
        pose = self.pose(s["kps_t"], s["hair"])
        if pose.shape[1] >= self.pose_channels:
            pose = pose[:, : self.pose_channels]
        else:
            pose = F.pad(pose, (0, 0, 0, 0, 0, self.pose_channels - pose.shape[1]))
        v = self.dit(
            x, t0, deltas, None, train=False,
            ref_latent=s["ref"], pose_map=pose.to(dt), keypoint_deltas=deltas,
            ref_face_latent=s["face"],
        )
        lat = x + (1.0 - w) * v
        img = self.dec(lat.contiguous(memory_format=torch.channels_last))
        s["lat_out"] = lat
        s["img_out"] = ((img.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
        s["bad_out"] = ~(torch.isfinite(lat).all() & torch.isfinite(img).all())

    def _slot(self, bsz: int) -> dict[str, Any]:
        s = self.slots.get(bsz)
        if s is not None:
            return s
        d, dt, n, c = self.device, self.dtype, self.input_size, self.in_channels
        s = {
            "kps_t": torch.zeros((bsz, NUM_KEYPOINTS, 4), device=d),
            "kps_r": torch.zeros((bsz, NUM_KEYPOINTS, 4), device=d),
            "hair": torch.zeros((bsz, 3, n, n), device=d),
            "ref": torch.zeros((bsz, c, n, n), device=d, dtype=dt),
            "face": torch.zeros((bsz, c, self.face_size, self.face_size), device=d, dtype=dt) if self.use_face else None,
            "last": torch.zeros((bsz, c, n, n), device=d, dtype=dt),
            "t0": torch.zeros((bsz,), device=d),
            "noise": self.noise1.expand(bsz, -1, -1, -1).contiguous(),
        }
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(3):  # compile / cudnn autotune happen here, outside capture
                self._body(s)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g, pool=self.pool):
            self._body(s)
        s["graph"] = g
        s["host"] = torch.empty(tuple(s["img_out"].shape), dtype=torch.uint8, pin_memory=True)
        s["host_bad"] = torch.empty((), dtype=torch.bool, pin_memory=True)
        self.slots[bsz] = s
        return s

    def prepare(self, bsz: int) -> None:
        self._slot(int(bsz))

    # ------------------------------------------------------------------ replay
    def _fill_ref(self, s: dict[str, Any], ref_latent: torch.Tensor, ref_face_latent: torch.Tensor | None) -> None:
        """Reference + face into the slot's static buffers, only when they changed.

        The engine swaps in a new tensor for a new reference and never writes into
        the old one; the version counter also catches an in-place write.
        """
        src = (ref_latent, ref_face_latent, _version(ref_latent), _version(ref_face_latent))
        old = s.get("ref_src")
        if old is not None and old[0] is src[0] and old[1] is src[1] and old[2:] == src[2:]:
            return
        bsz = s["ref"].shape[0]
        s["ref"].copy_(ref_latent[-1:].expand(bsz, -1, -1, -1))
        self._fill_face(s, ref_latent, ref_face_latent)
        s["ref_src"] = src  # holds the tensors, so their ids cannot be reused

    def _fill_face(self, s: dict[str, Any], ref_latent: torch.Tensor, ref_face_latent: torch.Tensor | None) -> None:
        """Face crop into its static buffer, sized like the DiT's ``ref_face_size``.

        Without a crop the DiT resizes the whole ref; do the same rather than replay
        the last character's face.
        """
        if s["face"] is None:
            return
        face = ref_face_latent[-1:] if ref_face_latent is not None else ref_latent[-1:]
        size = (self.face_size, self.face_size)
        if face.shape[-2:] != size:
            face = F.interpolate(face.float(), size=size, mode="bilinear", align_corners=False)
        s["face"].copy_(face.expand(s["face"].shape[0], -1, -1, -1))

    def run(
        self,
        *,
        kps_target: np.ndarray,
        kps_ref: np.ndarray,
        ref_latent: torch.Tensor,
        ref_face_latent: torch.Tensor | None,
        hair_maps: np.ndarray | torch.Tensor | None,
        last_latent: torch.Tensor | None,
        start_t: float,
    ) -> tuple[torch.Tensor, np.ndarray, bool]:
        """Returns (latents (B,C,H,W) clone in fp32, uint8 images (B,H,W,3), finite).

        ``finite`` is False when the latents or the decoded picture hold NaN/Inf.
        """
        # Static buffers are inference tensors (captured under inference_mode).
        with torch.inference_mode():
            return self._run(kps_target, kps_ref, ref_latent, ref_face_latent, hair_maps, last_latent, start_t)

    def _run(self, kps_target, kps_ref, ref_latent, ref_face_latent, hair_maps, last_latent, start_t):
        kt = np.asarray(kps_target, dtype=np.float32)
        if kt.ndim == 2:
            kt = kt[None]
        bsz = int(kt.shape[0])
        s = self._slot(bsz)
        d, n = self.device, self.input_size
        s["kps_t"].copy_(torch.from_numpy(kt), non_blocking=True)
        kr = np.asarray(kps_ref, dtype=np.float32)
        kr = kr[-1] if kr.ndim == 3 else kr
        s["kps_r"].copy_(torch.from_numpy(kr)[None].expand(bsz, -1, -1), non_blocking=True)
        self._fill_ref(s, ref_latent, ref_face_latent)
        if hair_maps is None:
            s["hair"].zero_()
        else:
            h = hair_maps if isinstance(hair_maps, torch.Tensor) else torch.from_numpy(np.asarray(hair_maps, np.float32))
            h = h.float()
            if h.ndim == 3:
                h = h[None]
            h = h.to(d, non_blocking=True)
            if h.shape[-2:] != (n, n):
                h = F.interpolate(h, size=(n, n), mode="bilinear", align_corners=False)
            s["hair"].copy_(h.expand(bsz, -1, -1, -1) if h.shape[0] == 1 else h)
        t0 = float(start_t) if last_latent is not None else 0.0
        t0 = min(max(t0, 0.0), 1.0 - 1e-4)
        if t0 > 0.0:
            s["last"].copy_(last_latent[-1:].expand(bsz, -1, -1, -1))
        s["t0"].fill_(t0)
        s["graph"].replay()
        lat = s["lat_out"].float().clone()
        s["host"].copy_(s["img_out"], non_blocking=True)
        s["host_bad"].copy_(s["bad_out"], non_blocking=True)
        self.done.record()
        self.done.synchronize()
        # Copied: the pinned buffer is overwritten next replay, and denoise_to_latents
        # hands the images out past _cuda_lock.
        return lat, s["host"].numpy().copy(), not bool(s["host_bad"])


def _version(t: torch.Tensor | None) -> int:
    """In-place write counter; -1 for None or an inference tensor (which has none)."""
    if t is None:
        return -1
    try:
        return int(t._version)
    except RuntimeError:
        return -1
