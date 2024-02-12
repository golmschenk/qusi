import torch
from torch import Tensor
from torch.nn import Linear, Module, Sigmoid


class SingleDenseLayerBinaryClassificationModel(Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.dense = Linear(in_features=input_size, out_features=1)
        self.activation = Sigmoid()

    def forward(self, x: Tensor):
        x = self.dense(x)
        x = self.activation(x)
        x = torch.reshape(x, (-1,))
        return x
