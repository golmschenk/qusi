from __future__ import annotations

import torch
from torch.nn import Module, Transformer, Conv1d, Parameter, Linear, Flatten, Sigmoid


class Torrin(Module):
    def __init__(self):
        super().__init__()
        embedding_size = 16
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=16, batch_first=True, num_decoder_layers=1)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=16, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, 3500])
        x = self.embedding_layer(x)
        x = torch.permute(x, (0, 2, 1))
        expanded_class_embedding = self.class_embedding.expand(x.size(0), -1, -1)
        x = torch.cat([expanded_class_embedding, x], dim=1)
        target = torch.zeros_like(x)
        x = self.transformer(x, target)
        x = x[:, 0, :]
        x = self.flatten(x)
        x = self.classification_layer(x)
        x = self.sigmoid(x)
        x = x.reshape([-1])
        return x


if __name__ == '__main__':
    model = Torrin()
    x_ = torch.rand(size=[7, 3500])
    y_ = model(x_)
    pass
