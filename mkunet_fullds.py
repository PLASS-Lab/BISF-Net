"""MK_UNet_Hybrid_DS_PraNet with FULL per-stage deep supervision (6 heads).

Identical to the E0 model (mkunet_hybrid_ds_pranet.py) EXCEPT it adds deep-supervision
heads on decoder stage 1 (dec1, c4 @H/8) and decoder stage 4 (dec4, c1 @H) — the two
decoder units that the 4-head PraNet-style E0 left unsupervised.

Heads (training):
  Sg  = bottleneck x5      (c5 @H/16)   [as E0]
  Sd1 = decoder1 output    (c4 @H/8)    [NEW]
  S5  = decoder2 output    (c3 @H/4)    [as E0]
  S4  = decoder3 output    (c2 @H/2)    [as E0]
  Sd4 = decoder4 output    (c1 @H)      [NEW]
  S3  = final (decoder5)   (c1 @H)      [as E0, finest / inference]
Training forward returns [S3, Sd4, S4, S5, Sd1, Sg] (finest first); inference returns [S3].
Use --ds-weights "1.0,1.0,1.0,1.0,1.0,1.0" (6 heads).
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

from funet.FusionUNet import FuseModule          # noqa: E402
from funet.utils import ConvBatchNorm, DownBlock  # noqa: E402


class MK_UNet_Hybrid_DS_PraNet_FullDS(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 3, base_channels: int = 32,
                 fusion_rounds: int = 2, expansion_factor: int = 2,
                 kernel_sizes: list[int] | None = None, **kwargs) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [1, 3, 5]
        c1 = base_channels; c2 = base_channels * 2; c3 = base_channels * 4
        c4 = base_channels * 8; c5 = base_channels * 8

        self.inc = ConvBatchNorm(in_channels, c1)
        self.down1 = DownBlock(c1, c2, nb_Conv=2)
        self.down2 = DownBlock(c2, c3, nb_Conv=2)
        self.down3 = DownBlock(c3, c4, nb_Conv=2)
        self.down4 = DownBlock(c4, c5, nb_Conv=2)

        self.fuse = FuseModule(base_channel=base_channels, nb_blocks=fusion_rounds)

        def dec(cin, cout):
            return mk_irb_bottleneck(cin, cout, 1, 1, expansion_factor=expansion_factor,
                                     dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder1 = dec(c5, c4)
        self.decoder2 = dec(c4, c3)
        self.decoder3 = dec(c3, c2)
        self.decoder4 = dec(c2, c1)
        self.decoder5 = dec(c1, c1)

        self.CA1 = ChannelAttention(c5, ratio=16)
        self.CA2 = ChannelAttention(c4, ratio=16)
        self.CA3 = ChannelAttention(c3, ratio=16)
        self.CA4 = ChannelAttention(c2, ratio=8)
        self.CA5 = ChannelAttention(c1, ratio=4)
        self.SA = SpatialAttention()

        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)       # S3 (final)
        self.sg_out = nn.Conv2d(c5, num_classes, kernel_size=1)    # Sg (bottleneck)
        self.ds_out0 = nn.Conv2d(c4, num_classes, kernel_size=1)   # Sd1 (decoder1)  NEW
        self.ds_out1 = nn.Conv2d(c3, num_classes, kernel_size=1)   # S5  (decoder2)
        self.ds_out2 = nn.Conv2d(c2, num_classes, kernel_size=1)   # S4  (decoder3)
        self.ds_out3 = nn.Conv2d(c1, num_classes, kernel_size=1)   # Sd4 (decoder4)  NEW
        self.return_all_heads = False   # eval-time flag: emit all 6 heads for inference fusion

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.float()
        input_size = x.shape[2:]
        emit = self.training or self.return_all_heads   # compute+return all heads when True

        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)

        if emit:
            sg = F.interpolate(self.sg_out(x5), size=input_size, mode="bilinear")

        x1, x2, x3, x4 = self.fuse(x1, x2, x3, x4)

        # stage 1
        out = self.CA1(x5) * x5; out = self.SA(out) * out
        dec1 = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode="bilinear"))
        if emit:
            sd1 = F.interpolate(self.ds_out0(dec1), size=input_size, mode="bilinear")
        out = torch.add(dec1, x4)
        # stage 2
        out = self.CA2(out) * out; out = self.SA(out) * out
        dec2 = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode="bilinear"))
        if emit:
            s5 = F.interpolate(self.ds_out1(dec2), size=input_size, mode="bilinear")
        out = torch.add(dec2, x3)
        # stage 3
        out = self.CA3(out) * out; out = self.SA(out) * out
        dec3 = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode="bilinear"))
        if emit:
            s4 = F.interpolate(self.ds_out2(dec3), size=input_size, mode="bilinear")
        out = torch.add(dec3, x2)
        # stage 4
        out = self.CA4(out) * out; out = self.SA(out) * out
        dec4 = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode="bilinear"))
        if emit:
            sd4 = F.interpolate(self.ds_out3(dec4), size=input_size, mode="bilinear")
        out = torch.add(dec4, x1)
        # stage 5
        out = self.CA5(out) * out; out = self.SA(out) * out
        out = F.relu(self.decoder5(out))
        s3 = self.out(out)

        if emit:
            return [s3, sd4, s4, s5, sd1, sg]
        return [s3]
