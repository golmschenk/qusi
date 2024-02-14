from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Dropout, Dropout1d, LeakyReLU, MaxPool1d, Module, Sigmoid
from typing_extensions import Self


class Hadryss(Module):
    def __init__(self, input_length: int):
        super().__init__()
        self.input_length: int = input_length
        pooling_sizes, dense_size = self.determine_block_pooling_sizes_and_dense_size()
        self.sigmoid = Sigmoid()
        self.block0 = LightCurveNetworkBlock(input_channels=1, output_channels=8, kernel_size=3,
                                             pooling_size=pooling_sizes[0])
        self.block1 = LightCurveNetworkBlock(input_channels=8, output_channels=8, kernel_size=3,
                                             pooling_size=pooling_sizes[1])
        self.block2 = LightCurveNetworkBlock(input_channels=8, output_channels=16, kernel_size=3,
                                             pooling_size=pooling_sizes[2],
                                             batch_normalization=True, dropout_rate=0.1)
        self.block3 = LightCurveNetworkBlock(input_channels=16, output_channels=32, kernel_size=3,
                                             pooling_size=pooling_sizes[3],
                                             batch_normalization=True, dropout_rate=0.1)
        self.block4 = LightCurveNetworkBlock(input_channels=32, output_channels=64, kernel_size=3,
                                             pooling_size=pooling_sizes[4],
                                             batch_normalization=True, dropout_rate=0.1)
        self.block5 = LightCurveNetworkBlock(input_channels=64, output_channels=128, kernel_size=3,
                                             pooling_size=pooling_sizes[5],
                                             batch_normalization=True, dropout_rate=0.1)
        self.block6 = LightCurveNetworkBlock(input_channels=128, output_channels=128, kernel_size=3,
                                             pooling_size=pooling_sizes[6],
                                             batch_normalization=True, dropout_rate=0.1)
        self.block7 = LightCurveNetworkBlock(input_channels=128, output_channels=128, kernel_size=3,
                                             pooling_size=pooling_sizes[7],
                                             batch_normalization=True, dropout_rate=0.1)
        self.block8 = LightCurveNetworkBlock(input_channels=128, output_channels=20, kernel_size=3,
                                             pooling_size=pooling_sizes[8], dropout_rate=0.1, spatial=False,
                                             length=dense_size + 2)
        self.block9 = LightCurveNetworkBlock(input_channels=20, output_channels=20, kernel_size=dense_size,
                                             pooling_size=1, dropout_rate=0.1)
        self.block10 = LightCurveNetworkBlock(input_channels=20, output_channels=20, kernel_size=1, pooling_size=1)
        self.prediction_layer = Conv1d(in_channels=20, out_channels=1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape([-1, 1, self.input_length])
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.prediction_layer(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (-1,))
        return x

    @classmethod
    def new(cls, input_length: int = 2500) -> Self:
        instance = cls(input_length=input_length)
        return instance

    def determine_block_pooling_sizes_and_dense_size(self) -> (list[int], int):
        number_of_pooling_blocks = 9
        pooling_sizes = [1] * number_of_pooling_blocks
        while True:
            for pooling_size_index, _pooling_size in enumerate(pooling_sizes):
                current_size = self.input_length
                for current_pooling_size in pooling_sizes:
                    current_size -= 2
                    current_size /= current_pooling_size
                    current_size = math.floor(current_size)
                if current_size <= 10:
                    return pooling_sizes, current_size
                pooling_sizes[pooling_size_index] += 1


class HadryssNonResizing(Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()
        self.block0 = LightCurveNetworkBlock(input_channels=1, output_channels=8, kernel_size=3, pooling_size=2)
        self.block1 = LightCurveNetworkBlock(input_channels=8, output_channels=8, kernel_size=3, pooling_size=2)
        self.block2 = LightCurveNetworkBlock(input_channels=8, output_channels=16, kernel_size=3, pooling_size=2,
                                             batch_normalization=True, dropout_rate=0.1)
        self.block3 = LightCurveNetworkBlock(input_channels=16, output_channels=32, kernel_size=3, pooling_size=2,
                                             batch_normalization=True, dropout_rate=0.1)
        self.block4 = LightCurveNetworkBlock(input_channels=32, output_channels=64, kernel_size=3, pooling_size=2,
                                             batch_normalization=True, dropout_rate=0.1)
        self.block5 = LightCurveNetworkBlock(input_channels=64, output_channels=128, kernel_size=3, pooling_size=2,
                                             batch_normalization=True, dropout_rate=0.1)
        self.block6 = LightCurveNetworkBlock(input_channels=128, output_channels=128, kernel_size=3, pooling_size=2,
                                             batch_normalization=True, dropout_rate=0.1)
        self.block7 = LightCurveNetworkBlock(input_channels=128, output_channels=128, kernel_size=3, pooling_size=2,
                                             batch_normalization=True, dropout_rate=0.1)
        self.block8 = LightCurveNetworkBlock(input_channels=128, output_channels=20, kernel_size=3, pooling_size=1,
                                             dropout_rate=0.1,
                                             spatial=False, length=7)
        self.block9 = LightCurveNetworkBlock(input_channels=20, output_channels=20, kernel_size=5, pooling_size=1,
                                             dropout_rate=0.1)
        self.block10 = LightCurveNetworkBlock(input_channels=20, output_channels=20, kernel_size=1, pooling_size=1)
        self.prediction_layer = Conv1d(in_channels=20, out_channels=1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape([-1, 1, 2500])
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.prediction_layer(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (-1,))
        return x


class LightCurveNetworkBlock(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, pooling_size: int,
                 dropout_rate: float = 0.0, batch_normalization: bool = False, spatial: bool = True,
                 length: int | None = None):
        super().__init__()
        self.leaky_relu = LeakyReLU()
        self.convolution = Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size)
        self.spatial: bool = spatial
        self.output_channels: int = output_channels
        if dropout_rate > 0:
            if spatial:
                self.dropout = Dropout1d(dropout_rate)
            else:
                self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
        if pooling_size > 1:
            self.max_pooling = MaxPool1d(kernel_size=pooling_size)
        else:
            self.max_pooling = None
        if batch_normalization:
            if spatial:
                self.batch_normalization = BatchNorm1d(num_features=output_channels)
            else:
                assert length is not None
                self.batch_normalization = BatchNorm1d(num_features=output_channels * length)
        else:
            self.batch_normalization = None

    def forward(self, x):
        x = self.convolution(x)
        x = self.leaky_relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.max_pooling is not None:
            x = self.max_pooling(x)
        if self.batch_normalization is not None:
            if not self.spatial:
                old_shape = x.shape
                x = torch.reshape(x, [-1, torch.prod(torch.tensor(old_shape[1:]))])
            x = self.batch_normalization(x)
            if not self.spatial:
                x = torch.reshape(x, old_shape)
        return x
