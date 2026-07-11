from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# MK-UNet decoder components
from mkunet_network import ChannelAttention, SpatialAttention, mk_irb_bottleneck

# FusionU-Net encoder and fusion components
_funet_root = str(Path(__file__).resolve().parent.parent / "FusionU-Net")
if _funet_root not in sys.path:
    sys.path.append(_funet_root)

from funet.FusionUNet import FuseModule
from funet.utils import ConvBatchNorm, DownBlock


class MK_UNet_Hybrid(nn.Module):
    """FusionUNet encoder + FusionUNet skip fusion + MK-UNet MKIR decoder.

    The FusionUNet encoder produces 4 skip features at doubling channel widths
    [C, 2C, 4C, 8C] and a bottleneck at 8C.  The FuseModule performs
    bidirectional cross-level fusion on those skips.  The decoder uses MK-UNet
    style MKIR bottleneck blocks with Channel- and Spatial-Attention, bilinear
    upsampling, and additive skip merging.

    With base_channels=32 the channel schedule is [32, 64, 128, 256, 256].
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        base_channels: int = 32,
        fusion_rounds: int = 2,
        expansion_factor: int = 2,
        kernel_sizes: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [1, 3, 5]

        c1 = base_channels            # 32
        c2 = base_channels * 2        # 64
        c3 = base_channels * 4        # 128
        c4 = base_channels * 8        # 256
        c5 = base_channels * 8        # 256 (bottleneck)

        # ---------- FusionUNet Encoder ----------
        self.inc = ConvBatchNorm(in_channels, c1)
        self.down1 = DownBlock(c1, c2, nb_Conv=2)
        self.down2 = DownBlock(c2, c3, nb_Conv=2)
        self.down3 = DownBlock(c3, c4, nb_Conv=2)
        self.down4 = DownBlock(c4, c5, nb_Conv=2)

        # ---------- FusionUNet Skip Fusion ----------
        self.fuse = FuseModule(base_channel=base_channels, nb_blocks=fusion_rounds)

        # ---------- MK-UNet MKIR Decoder ----------
        # 4 stages with upsampling to match FusionUNet's 4 downsampling steps
        self.decoder1 = mk_irb_bottleneck(
            c5, c4, 1, 1,
            expansion_factor=expansion_factor, dw_parallel=True, add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder2 = mk_irb_bottleneck(
            c4, c3, 1, 1,
            expansion_factor=expansion_factor, dw_parallel=True, add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder3 = mk_irb_bottleneck(
            c3, c2, 1, 1,
            expansion_factor=expansion_factor, dw_parallel=True, add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder4 = mk_irb_bottleneck(
            c2, c1, 1, 1,
            expansion_factor=expansion_factor, dw_parallel=True, add=True,
            kernel_sizes=kernel_sizes,
        )
        # Final refinement at full resolution (no upsampling needed)
        self.decoder5 = mk_irb_bottleneck(
            c1, c1, 1, 1,
            expansion_factor=expansion_factor, dw_parallel=True, add=True,
            kernel_sizes=kernel_sizes,
        )

        self.CA1 = ChannelAttention(c5, ratio=16)
        self.CA2 = ChannelAttention(c4, ratio=16)
        self.CA3 = ChannelAttention(c3, ratio=16)
        self.CA4 = ChannelAttention(c2, ratio=8)
        self.CA5 = ChannelAttention(c1, ratio=4)

        self.SA = SpatialAttention()

        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.float()

        # ---------- Encoder ----------
        x1 = self.inc(x)          # (B, c1, H,   W)
        x2 = self.down1(x1)       # (B, c2, H/2, W/2)
        x3 = self.down2(x2)       # (B, c3, H/4, W/4)
        x4 = self.down3(x3)       # (B, c4, H/8, W/8)
        x5 = self.down4(x4)       # (B, c5, H/16, W/16)

        # ---------- Skip Fusion ----------
        x1, x2, x3, x4 = self.fuse(x1, x2, x3, x4)

        # ---------- Decoder ----------
        out = self.CA1(x5) * x5
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, x4)

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, x3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, x2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, x1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(self.decoder5(out))

        return [self.out(out)]
