from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Linear, Module, Sigmoid
from typing_extensions import Self


class SingleDenseLayerBinaryClassificationModel(Module):
    def __init__(self, input_size: int, activation: Module):
        super().__init__()
        self.dense = Linear(in_features=input_size, out_features=1)
        self.activation = activation

    def forward(self, x: Tensor):
        x = self.dense(x)
        x = self.activation(x)
        x = torch.reshape(x, (-1,))
        return x

    @classmethod
    def new(cls, input_size: int, activation: Module | None = None) -> Self:
        if activation is None:
            activation = Sigmoid()
        instance = cls(input_size=input_size, activation=activation)
        return instance
