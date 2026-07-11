"""MK_UNet_Hybrid_DS with PraNet-style deep supervision (Sg, S5, S4, S3).

Sg = bottleneck global map (coarsest)
S5 = decoder stage 2 tap at H/4
S4 = decoder stage 3 tap at H/2
S3 = final main output (finest = inference prediction)

All outputs upsampled to input resolution during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mkunet_network import ChannelAttention, SpatialAttention, mk_irb_bottleneck

_funet_root = str(Path(__file__).resolve().parent.parent / "FusionU-Net")
if _funet_root not in sys.path:
    sys.path.append(_funet_root)

from funet.FusionUNet import FuseModule
from funet.utils import ConvBatchNorm, DownBlock


class MK_UNet_Hybrid_DS_PraNet(nn.Module):
    """FusionUNet encoder + FusionUNet fusion + MK-UNet decoder.

    PraNet-style deep supervision with 4 outputs:
      Training:  [S3, S4, S5, Sg]  (finest → coarsest, all at input resolution)
      Inference: [S3]
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
        c5 = base_channels * 8        # 256

        # ---------- FusionUNet Encoder ----------
        self.inc = ConvBatchNorm(in_channels, c1)
        self.down1 = DownBlock(c1, c2, nb_Conv=2)
        self.down2 = DownBlock(c2, c3, nb_Conv=2)
        self.down3 = DownBlock(c3, c4, nb_Conv=2)
        self.down4 = DownBlock(c4, c5, nb_Conv=2)

        # ---------- FusionUNet Skip Fusion ----------
        self.fuse = FuseModule(base_channel=base_channels, nb_blocks=fusion_rounds)

        # ---------- MK-UNet MKIR Decoder ----------
        self.decoder1 = mk_irb_bottleneck(
            c5, c4, 1, 1, expansion_factor=expansion_factor,
            dw_parallel=True, add=True, kernel_sizes=kernel_sizes,
        )
        self.decoder2 = mk_irb_bottleneck(
            c4, c3, 1, 1, expansion_factor=expansion_factor,
            dw_parallel=True, add=True, kernel_sizes=kernel_sizes,
        )
        self.decoder3 = mk_irb_bottleneck(
            c3, c2, 1, 1, expansion_factor=expansion_factor,
            dw_parallel=True, add=True, kernel_sizes=kernel_sizes,
        )
        self.decoder4 = mk_irb_bottleneck(
            c2, c1, 1, 1, expansion_factor=expansion_factor,
            dw_parallel=True, add=True, kernel_sizes=kernel_sizes,
        )
        self.decoder5 = mk_irb_bottleneck(
            c1, c1, 1, 1, expansion_factor=expansion_factor,
            dw_parallel=True, add=True, kernel_sizes=kernel_sizes,
        )

        self.CA1 = ChannelAttention(c5, ratio=16)
        self.CA2 = ChannelAttention(c4, ratio=16)
        self.CA3 = ChannelAttention(c3, ratio=16)
        self.CA4 = ChannelAttention(c2, ratio=8)
        self.CA5 = ChannelAttention(c1, ratio=4)

        self.SA = SpatialAttention()

        # ---------- Output heads ----------
        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)       # S3 (final)

        # Deep supervision heads (PraNet naming)
        self.ds_out1 = nn.Conv2d(c3, num_classes, kernel_size=1)   # S5: 128ch at H/4
        self.ds_out2 = nn.Conv2d(c2, num_classes, kernel_size=1)   # S4: 64ch at H/2
        self.sg_out = nn.Conv2d(c5, num_classes, kernel_size=1)    # Sg: 256ch at H/16

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.float()
        input_size = x.shape[2:]  # (H, W)

        # ---------- Encoder ----------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ---------- Sg: global map from bottleneck ----------
        if self.training:
            sg = F.interpolate(self.sg_out(x5), size=input_size, mode="bilinear")

        # ---------- Skip Fusion ----------
        x1, x2, x3, x4 = self.fuse(x1, x2, x3, x4)

        # ---------- Decoder stage 1 (256→256, H/16→H/8) ----------
        out = self.CA1(x5) * x5
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, x4)

        # ---------- Decoder stage 2 (256→128, H/8→H/4) ----------
        out = self.CA2(out) * out
        out = self.SA(out) * out
        dec2 = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode="bilinear"))
        if self.training:
            s5 = F.interpolate(self.ds_out1(dec2), size=input_size, mode="bilinear")
        out = torch.add(dec2, x3)

        # ---------- Decoder stage 3 (128→64, H/4→H/2) ----------
        out = self.CA3(out) * out
        out = self.SA(out) * out
        dec3 = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode="bilinear"))
        if self.training:
            s4 = F.interpolate(self.ds_out2(dec3), size=input_size, mode="bilinear")
        out = torch.add(dec3, x2)

        # ---------- Decoder stage 4 (64→32, H/2→H) ----------
        out = self.CA4(out) * out
        out = self.SA(out) * out
        dec4 = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(dec4, x1)

        # ---------- Decoder stage 5 (32→32, refinement at H) ----------
        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(self.decoder5(out))
        s3 = self.out(out)  # S3 = final prediction

        if self.training:
            return [s3, s4, s5, sg]
        return [s3]
