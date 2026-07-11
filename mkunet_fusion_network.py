from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mkunet_network import ChannelAttention, SpatialAttention, mk_irb_bottleneck


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECA(nn.Module):
    def __init__(self, channel: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class PixelShuffleUp(nn.Module):
    """Learnable 2x upsampling via sub-pixel convolution (pixel shuffle)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 4)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shuffle(self.bn(self.conv(x))))


def reshape_downsample(x: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Expected even spatial size for reshape_downsample, got {(height, width)}")

    reshaped = x.new_empty((batch_size, channels * 4, height // 2, width // 2))
    reshaped[:, 0::4, :, :] = x[:, :, 0::2, 0::2]
    reshaped[:, 1::4, :, :] = x[:, :, 0::2, 1::2]
    reshaped[:, 2::4, :, :] = x[:, :, 1::2, 0::2]
    reshaped[:, 3::4, :, :] = x[:, :, 1::2, 1::2]
    return reshaped


def reshape_upsample(x: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    if channels % 4 != 0:
        raise ValueError(f"Expected channel count divisible by 4 for reshape_upsample, got {channels}")

    reshaped = x.new_empty((batch_size, channels // 4, height * 2, width * 2))
    reshaped[:, :, 0::2, 0::2] = x[:, 0::4, :, :]
    reshaped[:, :, 0::2, 1::2] = x[:, 1::4, :, :]
    reshaped[:, :, 1::2, 0::2] = x[:, 2::4, :, :]
    reshaped[:, :, 1::2, 1::2] = x[:, 3::4, :, :]
    return reshaped


class DownFuseBlock(nn.Module):
    def __init__(self, shallow_channels: int, deep_channels: int) -> None:
        super().__init__()
        intermediate_channels = shallow_channels * 2
        self.down = reshape_downsample
        self.local_conv = nn.Conv2d(
            shallow_channels * 4,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=shallow_channels,
            bias=False,
        )
        self.local_norm = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        if intermediate_channels == deep_channels:
            self.match = nn.Identity()
        else:
            self.match = nn.Sequential(
                nn.Conv2d(intermediate_channels, deep_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(deep_channels),
                nn.ReLU(inplace=True),
            )
        self.fuse_conv = ConvBNAct(deep_channels, deep_channels)
        self.eca = ECA(deep_channels)

    def forward(self, shallow_feature: torch.Tensor, deep_feature: torch.Tensor) -> torch.Tensor:
        downsampled = self.down(shallow_feature)
        downsampled = self.relu(self.local_norm(self.local_conv(downsampled)))
        downsampled = self.match(downsampled)
        fused = self.fuse_conv(deep_feature * 0.75 + downsampled * 0.25) + deep_feature
        return self.eca(fused)


class UpFuseBlock(nn.Module):
    def __init__(self, shallow_channels: int, deep_channels: int) -> None:
        super().__init__()
        if deep_channels % 4 != 0:
            raise ValueError(f"Expected deep_channels divisible by 4, got {deep_channels}")

        reshaped_channels = deep_channels // 4
        intermediate_channels = deep_channels // 2
        self.up = reshape_upsample
        self.local_conv = nn.Conv2d(
            reshaped_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=reshaped_channels,
            bias=False,
        )
        self.local_norm = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        if intermediate_channels == shallow_channels:
            self.match = nn.Identity()
        else:
            self.match = nn.Sequential(
                nn.Conv2d(intermediate_channels, shallow_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(shallow_channels),
                nn.ReLU(inplace=True),
            )
        self.fuse_conv = ConvBNAct(shallow_channels, shallow_channels)
        self.eca = ECA(shallow_channels)

    def forward(self, shallow_feature: torch.Tensor, deep_feature: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(deep_feature)
        upsampled = self.relu(self.local_norm(self.local_conv(upsampled)))
        upsampled = self.match(upsampled)
        fused = self.fuse_conv(shallow_feature * 0.75 + upsampled * 0.25) + shallow_feature
        return self.eca(fused)


class FeatureFusionBlock(nn.Module):
    def __init__(self, skip_channels: list[int]) -> None:
        super().__init__()
        if len(skip_channels) != 4:
            raise ValueError(f"Expected four skip channel widths, got {skip_channels}")

        c1, c2, c3, c4 = skip_channels
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)
        self.norm3 = nn.BatchNorm2d(c3)
        self.norm4 = nn.BatchNorm2d(c4)

        self.down1 = DownFuseBlock(c1, c2)
        self.down2 = DownFuseBlock(c2, c3)
        self.down3 = DownFuseBlock(c3, c4)

        self.up3 = UpFuseBlock(c3, c4)
        self.up2 = UpFuseBlock(c2, c3)
        self.up1 = UpFuseBlock(c1, c2)

    def forward(
        self,
        fp1: torch.Tensor,
        fp2: torch.Tensor,
        fp3: torch.Tensor,
        fp4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fp1 = self.norm1(fp1)
        fp2 = self.norm2(fp2)
        fp3 = self.norm3(fp3)
        fp4 = self.norm4(fp4)

        fp2 = self.down1(fp1, fp2)
        fp3 = self.down2(fp2, fp3)
        fp4 = self.down3(fp3, fp4)

        fp3 = self.up3(fp3, fp4)
        fp2 = self.up2(fp2, fp3)
        fp1 = self.up1(fp1, fp2)
        return fp1, fp2, fp3, fp4


class FeatureFusionModule(nn.Module):
    def __init__(self, skip_channels: list[int], num_rounds: int = 2) -> None:
        super().__init__()
        num_rounds = max(1, num_rounds)
        self.blocks = nn.ModuleList(FeatureFusionBlock(skip_channels) for _ in range(num_rounds))

    def forward(
        self,
        fp1: torch.Tensor,
        fp2: torch.Tensor,
        fp3: torch.Tensor,
        fp4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            fp1, fp2, fp3, fp4 = block(fp1, fp2, fp3, fp4)
        return fp1, fp2, fp3, fp4


class MK_UNet_FFM(nn.Module):
    """MK-UNet with FusionU-Net-style bidirectional skip fusion replacing GAG."""

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        channels: list[int] | None = None,
        depths: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        expansion_factor: int = 2,
        fusion_rounds: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 192, 320]
        depths = depths or [2, 2, 2, 2, 2]
        kernel_sizes = kernel_sizes or [1, 3, 5]

        self.encoder1 = mk_irb_bottleneck(
            in_channels,
            channels[0],
            depths[0],
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.encoder2 = mk_irb_bottleneck(
            channels[0],
            channels[1],
            depths[1],
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.encoder3 = mk_irb_bottleneck(
            channels[1],
            channels[2],
            depths[2],
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.encoder4 = mk_irb_bottleneck(
            channels[2],
            channels[3],
            depths[3],
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.encoder5 = mk_irb_bottleneck(
            channels[3],
            channels[4],
            depths[4],
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )

        # This replaces the original GAG skip refinement path (AG1-AG4).
        self.skip_fusion = FeatureFusionModule(skip_channels=channels[:4], num_rounds=fusion_rounds)

        self.decoder1 = mk_irb_bottleneck(
            channels[4],
            channels[3],
            1,
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder2 = mk_irb_bottleneck(
            channels[3],
            channels[2],
            1,
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder3 = mk_irb_bottleneck(
            channels[2],
            channels[1],
            1,
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder4 = mk_irb_bottleneck(
            channels[1],
            channels[0],
            1,
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )
        self.decoder5 = mk_irb_bottleneck(
            channels[0],
            channels[0],
            1,
            1,
            expansion_factor=expansion_factor,
            dw_parallel=True,
            add=True,
            kernel_sizes=kernel_sizes,
        )

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        self.up1 = PixelShuffleUp(channels[3])
        self.up2 = PixelShuffleUp(channels[2])
        self.up3 = PixelShuffleUp(channels[1])
        self.up4 = PixelShuffleUp(channels[0])
        self.up5 = PixelShuffleUp(channels[0])

        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out

        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out

        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out

        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        out = F.max_pool2d(self.encoder5(out), 2, 2)

        # Fused skip tensors replace the original AG1-AG4 calls.
        t1, t2, t3, t4 = self.skip_fusion(t1, t2, t3, t4)

        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = self.up1(self.decoder1(out))
        out = torch.add(out, t4)

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = self.up2(self.decoder2(out))
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = self.up3(self.decoder3(out))
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = self.up4(self.decoder4(out))
        out = torch.add(out, t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = self.up5(self.decoder5(out))
        p4 = self.out4(out)
        return [p4]
