"""HRNetV2 implementation for anime face landmark detection.

Pure PyTorch implementation without mmpose dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for HRNet."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    """High resolution module for HRNet."""

    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
    ):
        super().__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = (
                f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})'
            )
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = (
                f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS({len(num_inchannels)})'
            )
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=0.1
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=0.1),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=0.1
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=0.1
                                    ),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNetBackbone(nn.Module):
    """HRNet backbone for pose estimation."""

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self, extra):
        super().__init__()
        self.extra = extra

        # Stem network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.stage1_cfg = extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block = self.blocks_dict[self.stage1_cfg['block']]
        num_blocks = self.stage1_cfg['num_blocks'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # Stage 2
        self.stage2_cfg = extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block = self.blocks_dict[self.stage2_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        # Stage 3
        self.stage3_cfg = extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block = self.blocks_dict[self.stage3_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        # Stage 4
        self.stage4_cfg = extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block = self.blocks_dict[self.stage4_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=0.1),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=0.1),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list


class HeatmapHead(nn.Module):
    """Heatmap head for keypoint detection matching mmpose implementation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_out_channels=(270,),
        conv_kernel_sizes=(1,),
    ):
        super().__init__()

        # Build conv layers if specified
        self.conv_layers = None
        if conv_out_channels:
            conv_layers = []
            for i, (out_c, kernel_size) in enumerate(
                zip(conv_out_channels, conv_kernel_sizes)
            ):
                in_c = in_channels if i == 0 else conv_out_channels[i - 1]
                padding = (kernel_size - 1) // 2
                conv_layers.extend(
                    [
                        nn.Conv2d(
                            in_c,
                            out_c,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                    ]
                )
            self.conv_layers = nn.Sequential(*conv_layers)
            final_in_channels = conv_out_channels[-1]
        else:
            final_in_channels = in_channels

        # Final layer to produce heatmaps
        self.final_layer = nn.Conv2d(
            final_in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        if self.conv_layers is not None:
            x = self.conv_layers(x)
        x = self.final_layer(x)
        return x


class HRNetV2(nn.Module):
    """HRNetV2 for anime face landmark detection.

    Args:
        num_keypoints: Number of keypoints to detect (default: 28)
        pretrained: Path to pretrained weights
    """

    def __init__(self, num_keypoints=28, pretrained=None):
        super().__init__()

        # HRNet configuration matching hrnetv2.py
        extra = {
            'stage1': {
                'num_modules': 1,
                'num_branches': 1,
                'block': 'BOTTLENECK',
                'num_blocks': [4],
                'num_channels': [64],
            },
            'stage2': {
                'num_modules': 1,
                'num_branches': 2,
                'block': 'BASIC',
                'num_blocks': [4, 4],
                'num_channels': [18, 36],
            },
            'stage3': {
                'num_modules': 4,
                'num_branches': 3,
                'block': 'BASIC',
                'num_blocks': [4, 4, 4],
                'num_channels': [18, 36, 72],
            },
            'stage4': {
                'num_modules': 3,
                'num_branches': 4,
                'block': 'BASIC',
                'num_blocks': [4, 4, 4, 4],
                'num_channels': [18, 36, 72, 144],
            },
        }

        self.backbone = HRNetBackbone(extra)

        # Feature map processor (concat all branches)
        # Channels: 18 + 36 + 72 + 144 = 270
        self.num_keypoints = num_keypoints
        # HeatmapHead matching mmpose config: conv_out_channels=(270,), conv_kernel_sizes=(1,)
        self.head = HeatmapHead(
            in_channels=270,
            out_channels=num_keypoints,
            conv_out_channels=(270,),
            conv_kernel_sizes=(1,),
        )

        # Data preprocessing parameters
        self.register_buffer(
            'mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        )

        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights from mmpose checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Extract state_dict if it's wrapped
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove 'module.' prefix and map keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith('module.'):
                k = k[7:]

            # Map keypoint_head to head
            if k.startswith('keypoint_head.final_layer.'):
                # keypoint_head.final_layer.0 -> head.conv_layers.0
                # keypoint_head.final_layer.1 -> head.conv_layers.1
                # keypoint_head.final_layer.3 -> head.final_layer
                parts = k.split('.')
                layer_idx = parts[2]  # '0', '1', '3'
                suffix = '.'.join(parts[3:])  # 'weight', 'bias', etc.

                if layer_idx == '3':
                    new_k = f'head.final_layer.{suffix}'
                else:
                    new_k = f'head.conv_layers.{layer_idx}.{suffix}'

                new_state_dict[new_k] = v
                continue

            new_state_dict[k] = v

        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(
            new_state_dict, strict=False
        )

        # Debug output (can be removed in production)
        if missing_keys:
            # Filter out mean/std buffers which are initialized in __init__
            real_missing = [k for k in missing_keys if k not in ['mean', 'std']]
            if real_missing:
                print(f'Warning: Missing keys in checkpoint: {real_missing[:5]}...')
        if unexpected_keys:
            print(f'Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...')

    def preprocess(self, x):
        """Preprocess input image.

        Args:
            x: Input tensor in BGR format, shape (B, 3, H, W), range [0, 255]

        Returns:
            Preprocessed tensor
        """
        # BGR to RGB
        x = x[:, [2, 1, 0], :, :]

        # Normalize
        x = (x - self.mean) / self.std

        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor, shape (B, 3, H, W)

        Returns:
            Heatmaps, shape (B, num_keypoints, H/4, W/4)
        """
        # Get multi-scale features from backbone
        y_list = self.backbone(x)

        # Upsample all branches to the same size (highest resolution)
        target_size = y_list[0].shape[2:]
        upsampled = []
        for i, y in enumerate(y_list):
            if i == 0:
                upsampled.append(y)
            else:
                upsampled.append(
                    F.interpolate(
                        y, size=target_size, mode='bilinear', align_corners=False
                    )
                )

        # Concatenate all branches
        x = torch.cat(upsampled, dim=1)

        # Generate heatmaps
        heatmaps = self.head(x)

        return heatmaps

    def decode_heatmaps(self, heatmaps):
        """Decode heatmaps to keypoint coordinates.

        Args:
            heatmaps: Heatmap tensor, shape (B, K, H, W)

        Returns:
            keypoints: Keypoint coordinates, shape (B, K, 2)
            scores: Keypoint scores, shape (B, K)
        """
        B, K, H, W = heatmaps.shape

        # Flatten heatmaps
        heatmaps_flat = heatmaps.view(B, K, -1)

        # Get max values and indices
        scores, indices = torch.max(heatmaps_flat, dim=2)

        # Convert indices to coordinates
        y = (indices // W).float()
        x = (indices % W).float()

        # Stack coordinates
        keypoints = torch.stack([x, y], dim=2)

        return keypoints, scores
