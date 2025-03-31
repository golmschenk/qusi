from __future__ import annotations

import math
from typing_extensions import Self

import torch
from torch import tensor, Tensor
from torch.nn import Module, Transformer, Conv1d, Parameter, Linear, Flatten, Sigmoid, Dropout


class TorrinBinaryClassEndModule(Module):
    """
    A module for the end of the Torrin model designed for binary classification.
    """

    def __init__(self):
        super().__init__()
        self.prediction_layer = Linear(in_features=100, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.prediction_layer(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (-1,))
        return x

    @classmethod
    def new(cls):
        return cls()


class TorrinMultiClassScoreEndModule(Module):
    """
    A module for the end of the Torrin model designed for multi classification without softmax.
    """

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.number_of_classes: int = number_of_classes
        self.prediction_layer = Linear(in_features=100, out_features=self.number_of_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.prediction_layer(x)
        x = torch.reshape(x, (-1, self.number_of_classes))
        return x

    @classmethod
    def new(cls, number_of_classes: int):
        return cls(number_of_classes)


class Torrin(Module):
    @classmethod
    def new(cls, input_length: int = 3500, end_module: Module | None = None) -> Self:
        if end_module is None:
            end_module = TorrinBinaryClassEndModule.new()
        return cls(input_length=input_length, end_module=end_module)

    def __init__(self, input_length: int, end_module: Module):
        super().__init__()
        embedding_size = 16
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=16, batch_first=True,
                                       num_decoder_layers=1)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.end_latent_layer = Linear(in_features=16, out_features=100)
        self.end_module = end_module
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
        x = self.embedding_layer(x)
        x = torch.permute(x, (0, 2, 1))
        expanded_class_embedding = self.class_embedding.expand(x.size(0), -1, -1)
        x = torch.cat([expanded_class_embedding, x], dim=1)
        target = torch.zeros_like(x)
        x = self.transformer(x, target)
        x = x[:, 0, :]
        x = self.flatten(x)
        x = self.end_latent_layer(x)
        x = self.end_module(x)
        return x


class PositionalEncoding(Module):
    def __init__(self, embedding_dimensions, dropout, input_length=100):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        positional_encoding = torch.zeros(input_length, embedding_dimensions)
        position = torch.arange(0, input_length).unsqueeze(1)
        inverse_denominator = torch.exp(
            torch.arange(0, embedding_dimensions, 2) * -(math.log(10000.0) / embedding_dimensions))
        phase = position * inverse_denominator
        positional_encoding[:, 0::2] = torch.sin(phase)
        positional_encoding[:, 1::2] = torch.cos(phase)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + tensor(self.positional_encoding[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Torrin1(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 32
        hidden_units = 16
        num_decoder_layers = 1
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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


class Torrin2(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 16
        hidden_units = 32
        num_decoder_layers = 1
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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


class Torrin3(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 32
        hidden_units = 32
        num_decoder_layers = 1
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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


class Torrin4(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 32
        hidden_units = 32
        num_decoder_layers = 2
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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


class Torrin5(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 512
        hidden_units = 2048
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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


class Torrin6(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 512
        hidden_units = 2048
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=1)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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


class Torrin7(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 32
        hidden_units = 32
        num_decoder_layers = 1
        self.input_length = input_length
        self.embedding_layer0 = Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=7)
        self.embedding_layer1 = Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=5)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
        x = self.embedding_layer0(x)
        x = self.embedding_layer1(x)
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


class Torrin8(Module):
    @classmethod
    def new(cls, input_length: int = 3500) -> Self:
        return cls(input_length=input_length)

    def __init__(self, input_length: int):
        super().__init__()
        embedding_size = 128
        hidden_units = 128
        num_decoder_layers = 1
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.classification_layer = Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
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

class Torrin9(Module):
    @classmethod
    def new(cls, input_length: int = 3500, end_module: Module | None = None) -> Self:
        if end_module is None:
            end_module = TorrinBinaryClassEndModule.new()
        return cls(input_length=input_length, end_module=end_module)

    def __init__(self, input_length: int, end_module: Module):
        super().__init__()
        embedding_size = 32
        hidden_units = 32
        num_decoder_layers = 1
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=0, input_length=100)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=hidden_units, batch_first=True,
                                       num_decoder_layers=num_decoder_layers)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.end_latent_layer = Linear(in_features=16, out_features=100)
        self.end_module = end_module
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
        x = self.embedding_layer(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.positional_encoding(x)
        expanded_class_embedding = self.class_embedding.expand(x.size(0), -1, -1)
        x = torch.cat([expanded_class_embedding, x], dim=1)
        target = torch.zeros_like(x)
        x = self.transformer(x, target)
        x = x[:, 0, :]
        x = self.flatten(x)
        x = self.end_latent_layer(x)
        x = self.end_module(x)
        return x


class Torrin10(Module):
    @classmethod
    def new(cls, input_length: int = 3500, end_module: Module | None = None) -> Self:
        if end_module is None:
            end_module = TorrinBinaryClassEndModule.new()
        return cls(input_length=input_length, end_module=end_module)

    def __init__(self, input_length: int, end_module: Module):
        super().__init__()
        embedding_size = 16
        self.input_length = input_length
        self.embedding_layer = Conv1d(in_channels=1, out_channels=embedding_size, kernel_size=35, stride=35)
        self.transformer = Transformer(d_model=embedding_size, dim_feedforward=16, batch_first=True,
                                       num_decoder_layers=1)
        self.class_embedding = Parameter(torch.randn([1, 1, embedding_size]))
        self.flatten = Flatten()
        self.end_latent_layer = Linear(in_features=16, out_features=100)
        self.end_module = end_module

    def forward(self, x):
        x = x.reshape([-1, 1, self.input_length])
        x = self.embedding_layer(x)
        x = torch.permute(x, (0, 2, 1))
        expanded_class_embedding = self.class_embedding.expand(x.size(0), -1, -1)
        x = torch.cat([expanded_class_embedding, x], dim=1)
        target = torch.zeros_like(x)
        x = self.transformer(x, target)
        x = x[:, 0, :]
        x = self.flatten(x)
        x = self.end_latent_layer(x)
        x = self.end_module(x)
        return x


if __name__ == '__main__':
    model = Torrin10.new()
    x_ = torch.rand(size=[7, 3500])
    y_ = model(x_)
    pass
