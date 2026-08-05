"""Hourglass mínima usada por el detector de bayas.

La jerarquía de atributos se conserva para que el ``state_dict`` del modelo
histórico de CircleNet siga siendo compatible.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class Convolution(nn.Module):
    def __init__(self, kernel_size: int, input_channels: int,
                 output_channels: int, stride: int = 1,
                 with_bn: bool = True) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            input_channels, output_channels, (kernel_size, kernel_size),
            padding=(padding, padding), stride=(stride, stride),
            bias=not with_bn,
        )
        self.bn = nn.BatchNorm2d(output_channels) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(tensor)))


class Residual(nn.Module):
    def __init__(self, _kernel_size: int, input_channels: int,
                 output_channels: int, stride: int = 1,
                 **_kwargs: object) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, (3, 3), padding=(1, 1),
            stride=(stride, stride), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, (3, 3), padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels, (1, 1),
                    stride=(stride, stride), bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
            if stride != 1 or input_channels != output_channels
            else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = self.skip(tensor)
        tensor = self.relu1(self.bn1(self.conv1(tensor)))
        tensor = self.bn2(self.conv2(tensor))
        return self.relu(tensor + residual)


LayerFactory = Callable[..., nn.Module]


def make_layer(kernel_size: int, input_channels: int, output_channels: int,
               modules: int, layer: LayerFactory = Convolution,
               **kwargs: object) -> nn.Sequential:
    layers = [layer(kernel_size, input_channels, output_channels, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(kernel_size, output_channels, output_channels, **kwargs))
    return nn.Sequential(*layers)


def make_reverse_layer(kernel_size: int, input_channels: int,
                       output_channels: int, modules: int,
                       layer: LayerFactory = Convolution,
                       **kwargs: object) -> nn.Sequential:
    layers = [
        layer(kernel_size, input_channels, input_channels, **kwargs)
        for _ in range(modules - 1)
    ]
    layers.append(layer(kernel_size, input_channels, output_channels, **kwargs))
    return nn.Sequential(*layers)


class MergeUp(nn.Module):
    def forward(self, first: torch.Tensor,
                second: torch.Tensor) -> torch.Tensor:
        return first + second


def make_pool_layer(_channels: int) -> nn.Sequential:
    # El downsampling ocurre en make_hourglass_layer, como en el modelo original.
    return nn.Sequential()


def make_unpool_layer(_channels: int) -> nn.Upsample:
    return nn.Upsample(scale_factor=2)


def make_head_layer(convolution_channels: int, current_channels: int,
                    output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        Convolution(3, convolution_channels, current_channels, with_bn=False),
        nn.Conv2d(current_channels, output_channels, (1, 1)),
    )


def make_inter_layer(channels: int) -> Residual:
    return Residual(3, channels, channels)


def make_convolution_layer(input_channels: int,
                           output_channels: int) -> Convolution:
    return Convolution(3, input_channels, output_channels)


def make_hourglass_layer(kernel_size: int, first_channels: int,
                         second_channels: int, modules: int,
                         layer: LayerFactory = Convolution,
                         **kwargs: object) -> nn.Sequential:
    layers = [
        layer(kernel_size, first_channels, second_channels, stride=2)
    ]
    layers.extend(
        layer(kernel_size, second_channels, second_channels, **kwargs)
        for _ in range(modules - 1)
    )
    return nn.Sequential(*layers)


class KeypointModule(nn.Module):
    def __init__(self, depth: int, dimensions: list[int],
                 module_counts: list[int],
                 layer: LayerFactory = Residual) -> None:
        super().__init__()
        current_modules = module_counts[0]
        next_modules = module_counts[1]
        current_channels = dimensions[0]
        next_channels = dimensions[1]

        self.n = depth
        self.up1 = make_layer(
            3, current_channels, current_channels, current_modules, layer=layer
        )
        self.max1 = make_pool_layer(current_channels)
        self.low1 = make_hourglass_layer(
            3, current_channels, next_channels, current_modules, layer=layer
        )
        if depth > 1:
            self.low2 = KeypointModule(
                depth - 1, dimensions[1:], module_counts[1:], layer=layer
            )
        else:
            self.low2 = make_layer(
                3, next_channels, next_channels, next_modules, layer=layer
            )
        self.low3 = make_reverse_layer(
            3, next_channels, current_channels, current_modules, layer=layer
        )
        self.up2 = make_unpool_layer(current_channels)
        self.merge = MergeUp()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        upper = self.up1(tensor)
        lower = self.low1(self.max1(tensor))
        lower = self.low2(lower)
        lower = self.up2(self.low3(lower))
        return self.merge(upper, lower)


class BerryHourglassNet(nn.Module):
    """Hourglass de dos stacks compatible con el checkpoint de producción."""

    def __init__(self, heads: dict[str, int], num_stacks: int = 2) -> None:
        super().__init__()
        depth = 4
        dimensions = [256, 256, 384, 384, 384, 512]
        module_counts = [2, 2, 2, 2, 2, 4]
        current_channels = dimensions[0]
        convolution_channels = 256

        self.nstack = num_stacks
        self.heads = heads
        self.pre = nn.Sequential(
            Convolution(7, 3, 128, stride=2),
            Residual(3, 128, 256, stride=2),
        )
        self.kps = nn.ModuleList([
            KeypointModule(depth, dimensions, module_counts, layer=Residual)
            for _ in range(num_stacks)
        ])
        self.cnvs = nn.ModuleList([
            make_convolution_layer(current_channels, convolution_channels)
            for _ in range(num_stacks)
        ])
        self.inters = nn.ModuleList([
            make_inter_layer(current_channels) for _ in range(num_stacks - 1)
        ])
        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(current_channels, current_channels, (1, 1), bias=False),
                nn.BatchNorm2d(current_channels),
            )
            for _ in range(num_stacks - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    convolution_channels, current_channels, (1, 1), bias=False
                ),
                nn.BatchNorm2d(current_channels),
            )
            for _ in range(num_stacks - 1)
        ])

        for head, output_channels in heads.items():
            modules = nn.ModuleList([
                make_head_layer(
                    convolution_channels, current_channels, output_channels
                )
                for _ in range(num_stacks)
            ])
            setattr(self, head, modules)
            if "hm" in head:
                for heatmap in modules:
                    heatmap[-1].bias.data.fill_(-2.19)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        intermediate = self.pre(image)
        outputs: list[dict[str, torch.Tensor]] = []
        for index in range(self.nstack):
            keypoints = self.kps[index](intermediate)
            convolution = self.cnvs[index](keypoints)
            output = {
                head: getattr(self, head)[index](convolution)
                for head in self.heads
            }
            outputs.append(output)
            if index < self.nstack - 1:
                intermediate = (
                    self.inters_[index](intermediate)
                    + self.cnvs_[index](convolution)
                )
                intermediate = self.inters[index](self.relu(intermediate))
        return outputs
