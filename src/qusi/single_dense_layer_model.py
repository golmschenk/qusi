import torch
from torch.nn import Module, Linear, Sigmoid


class SingleDenseLayerBinaryClassificationModel(Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.dense = Linear(in_features=input_size, out_features=1)
        self.activation = Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = torch.reshape(x, (-1,))
        return x
