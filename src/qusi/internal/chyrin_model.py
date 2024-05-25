from __future__ import annotations

import math

from torch import permute
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout1d,
    LeakyReLU,
    MaxPool1d,
    Module,
    ModuleList, ConstantPad1d, Sigmoid,
)


class Chyrin(Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleList()
        self.activation = LeakyReLU()
        self.sigmoid = Sigmoid()
        output_channels = 10
        self.blocks.append(ResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=1, dropout_rate=0.0,
            batch_normalization=False))
        input_channels = output_channels
        for output_channels in [10, 10, 20, 20, 30, 30, 40, 40, 50, 50]:
            self.blocks.append(ResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_scale_factor=2,
                dropout_rate=0.0,
                batch_normalization=False))
            input_channels = output_channels
            for _ in range(1):
                self.blocks.append(ResidualLightCurveNetworkBlock(
                    input_channels=input_channels, output_channels=output_channels, dropout_rate=0.0,
                    batch_normalization=False))
                input_channels = output_channels
        self.end_conv = Conv1d(input_channels, 1, kernel_size=3)

    def forward(self, x):
        x = x.reshape([-1, 1, 3500])
        for index, block in enumerate(self.blocks):
            x = block(x)
        x = self.end_conv(x)
        x = self.sigmoid(x)
        outputs = x.reshape([-1])
        return outputs


class ResidualLightCurveNetworkBlock(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 pooling_scale_factor: int = 1, batch_normalization: bool = False, dropout_rate: float = 0.0,
                 renorm: bool = False):
        super().__init__()
        self.activation = LeakyReLU()
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNorm1d(num_features=input_channels, track_running_stats=renorm)
        else:
            self.batch_normalization = None
        reduced_channels = output_channels // dimension_decrease_factor
        self.dimension_decrease_layer = Conv1d(
            in_channels=input_channels, out_channels=reduced_channels, kernel_size=1)
        self.convolutional_layer = Conv1d(
            in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=kernel_size,
            padding=math.floor(kernel_size / 2)
        )
        self.dimension_increase_layer = Conv1d(
            in_channels=reduced_channels, out_channels=output_channels, kernel_size=1)
        if pooling_scale_factor > 1:
            self.pooling_layer = MaxPool1d(kernel_size=pooling_scale_factor)
        else:
            self.pooling_layer = None
        self.input_to_output_channel_difference = input_channels - output_channels
        if output_channels != input_channels:
            if output_channels < input_channels:
                self.output_channels = output_channels
            else:
                self.dimension_change_layer = ConstantPad1d(padding=(0, -self.input_to_output_channel_difference),
                                                            value=0)
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = Dropout1d(p=dropout_rate)
        else:
            self.dropout_layer = None

    def forward(self, x):
        """
        The forward pass of the block.

        :param x: The input tensor.
        :return: The output tensor of the layer.
        """
        y = x
        if self.batch_normalization is not None:
            y = self.batch_normalization(y)
        y = self.dimension_decrease_layer(y)
        y = self.activation(y)
        y = self.convolutional_layer(y)
        y = self.activation(y)
        y = self.dimension_increase_layer(y)
        y = self.activation(y)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x)
            y = self.pooling_layer(y)
        if self.input_to_output_channel_difference != 0:
            x = permute(x, (0, 2, 1))
            if self.input_to_output_channel_difference < 0:
                x = self.dimension_change_layer(x)
            else:
                x = x[:, :, 0:self.output_channels]
            x = permute(x, (0, 2, 1))
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return x + y
