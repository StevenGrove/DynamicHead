import math
import torch
import torch.nn.functional as F
from torch import nn
from prodict import Prodict
from .wrappers import Conv2d
from .batch_norm import get_activation, get_norm


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride=1, 
        norm="BN", 
        activation=None
    ):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = out + shortcut
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    def __init__(
        self, 
        in_channels : int,
        out_channels : int,
        stride : int = 1,
        norm: str = "GN",
    ):
        super(Bottleneck, self).__init__()
        self.bottleneck = BasicBlock(in_channels,
                                     out_channels,
                                     stride=stride,
                                     norm=norm,
                                     activation=Prodict(NAME="ReLU", INPLACE=True))

        self.init_parameters()

    def init_parameters(self):
        for layer in self.bottleneck.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, input):
        output = self.bottleneck(input)
        return output


class ScaleConv2d(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        resize_method: str = "bilinear",
        norm: str = "GN",
        depthwise: bool = True
    ):
        super(ScaleConv2d, self).__init__()
        if depthwise:
            assert in_channels == out_channels
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        convs = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    groups=out_channels if depthwise else 1,
                ),
                get_norm(norm, out_channels)
            ) for _ in range(num_adjacent_scales)]
        self.convs = nn.ModuleList(convs)
        if resize_method == "bilinear":
            self.resize = lambda x, s : F.interpolate(
                x, size=s, mode="bilinear", align_corners=True)
        else:
            raise NotImplementedError()
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

    def init_parameters(self):
        for layer in self.convs.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, inputs):
        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x) for m in self.convs])
        
        outputs = []
        for l, x in enumerate(inputs):
            scale_feature = []
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-2:])
                    scale_feature.append(feature)

            scale_feature = sum(scale_feature) + x * self.output_weight
            if self.depth_module is not None:
                scale_feature = self.depth_module(scale_feature)
            outputs.append(scale_feature)

        return outputs
