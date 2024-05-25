from __future__ import annotations

from torch.nn import (
    LeakyReLU,
    Module,
    Sigmoid, Linear,
)


class SimpleDense(Module):
    def __init__(self):
        super().__init__()
        self.activation = LeakyReLU()
        self.sigmoid = Sigmoid()
        self.dense0 = Linear(in_features=3500, out_features=100)
        self.dense1 = Linear(in_features=100, out_features=100)
        self.dense2 = Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = x.reshape([-1, 3500])
        x = self.dense0(x)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.sigmoid(x)
        outputs = x.reshape([-1])
        return outputs
