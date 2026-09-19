"""HRNetV2-W18 that loads mmpose_anime-face_hrnetv2.pth (28 heatmaps)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(inn: int, out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inn, out, 3, stride, 1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inn: int, mid: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inn, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = _conv3x3(mid, mid, stride)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, mid * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BasicBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = _conv3x3(ch, ch)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = _conv3x3(ch, ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class HighResolutionModule(nn.Module):
    def __init__(self, channels: list[int], *, multi_scale: bool = True):
        super().__init__()
        self.channels = channels
        self.multi_scale = multi_scale
        self.branches = nn.ModuleList(
            [nn.Sequential(*[BasicBlock(ch) for _ in range(4)]) for ch in channels]
        )
        self.fuse_layers = self._make_fuse(channels)
        self.relu = nn.ReLU(inplace=True)

    def _make_fuse(self, channels: list[int]) -> nn.ModuleList:
        n = len(channels)
        fuses = nn.ModuleList()
        for j in range(n):
            row = nn.ModuleList()
            for i in range(n):
                if i == j:
                    row.append(nn.Identity())
                elif i > j:
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(channels[i], channels[j], 1, bias=False),
                            nn.BatchNorm2d(channels[j]),
                        )
                    )
                else:
                    steps = []
                    inn = channels[i]
                    for k in range(j - i):
                        last = k == j - i - 1
                        out = channels[j] if last else inn
                        block = [
                            _conv3x3(inn, out, 2),
                            nn.BatchNorm2d(out),
                        ]
                        if not last:
                            block.append(nn.ReLU(inplace=True))
                        steps.append(nn.Sequential(*block))
                    row.append(nn.Sequential(*steps))
            fuses.append(row)
        return fuses

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        n = len(self.channels)
        outs: list[torch.Tensor] = []
        for j in range(n if self.multi_scale else 1):
            y = xs[j]
            for i in range(n):
                if i == j:
                    continue
                feat = self.fuse_layers[j][i](xs[i])
                if i > j:
                    feat = F.interpolate(
                        feat, size=xs[j].shape[2:], mode="nearest"
                    )
                y = y + feat
            outs.append(self.relu(y))
        return outs if self.multi_scale else outs


class HRNetV2(nn.Module):
    """mmpose HRNetV2-W18 + 28-keypoint head."""

    def __init__(self, num_keypoints: int = 28):
        super().__init__()
        self.conv1 = _conv3x3(3, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = _conv3x3(64, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer1()
        self.transition1 = self._make_transition(256, [18, 36], from_single=True)
        self.stage2 = nn.Sequential(HighResolutionModule([18, 36]))
        self.transition2 = self._make_transition([18, 36], [18, 36, 72])
        self.stage3 = nn.Sequential(
            *[HighResolutionModule([18, 36, 72]) for _ in range(4)]
        )
        self.transition3 = self._make_transition([18, 36, 72], [18, 36, 72, 144])
        self.stage4 = nn.Sequential(
            HighResolutionModule([18, 36, 72, 144]),
            HighResolutionModule([18, 36, 72, 144]),
            HighResolutionModule([18, 36, 72, 144], multi_scale=True),
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(270, 270, 1),
            nn.BatchNorm2d(270),
            nn.ReLU(inplace=True),
            nn.Conv2d(270, num_keypoints, 1),
        )

    def _make_layer1(self) -> nn.Sequential:
        down = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        blocks = [Bottleneck(64, 64, downsample=down)]
        for _ in range(3):
            blocks.append(Bottleneck(256, 64))
        return nn.Sequential(*blocks)

    def _make_transition(
        self,
        inn: int | list[int],
        out: list[int],
        *,
        from_single: bool = False,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        if from_single:
            assert isinstance(inn, int)
            for i, ch in enumerate(out):
                if i == 0:
                    layers.append(
                        nn.Sequential(
                            _conv3x3(inn, ch),
                            nn.BatchNorm2d(ch),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    layers.append(
                        nn.Sequential(
                            nn.Sequential(
                                _conv3x3(inn, ch, 2),
                                nn.BatchNorm2d(ch),
                                nn.ReLU(inplace=True),
                            )
                        )
                    )
            return layers

        assert isinstance(inn, list)
        for i, ch in enumerate(out):
            if i < len(inn):
                layers.append(nn.Identity())
            else:
                layers.append(
                    nn.Sequential(
                        nn.Sequential(
                            _conv3x3(inn[-1], ch, 2),
                            nn.BatchNorm2d(ch),
                            nn.ReLU(inplace=True),
                        )
                    )
                )
        return layers

    def _apply_transition(
        self, xs: list[torch.Tensor], trans: nn.ModuleList
    ) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        for i, layer in enumerate(trans):
            if i < len(xs) and isinstance(layer, nn.Identity):
                outs.append(xs[i])
            elif i < len(xs):
                outs.append(layer(xs[0] if len(xs) == 1 else xs[i]))
            else:
                outs.append(layer(xs[-1]))
        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        y = self._apply_transition([x], self.transition1)
        y = self.stage2(y)
        y = self._apply_transition(y, self.transition2)
        y = self.stage3(y)
        y = self._apply_transition(y, self.transition3)
        y = self.stage4(y)
        size = y[0].shape[2:]
        fused = torch.cat(
            [y[0]]
            + [F.interpolate(t, size=size, mode="bilinear", align_corners=False) for t in y[1:]],
            dim=1,
        )
        return self.final_layer(fused)
