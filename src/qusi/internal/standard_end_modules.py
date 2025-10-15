import torch
from torch import Tensor
from torch.nn import Module, Conv1d, Sigmoid, Softmax


class BinaryClassEndModule(Module):
    """
    The standard end module for binary classification.
    """
    def __init__(self):
        super().__init__()
        self.prediction_layer = Conv1d(in_channels=100, out_channels=1, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.prediction_layer(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (-1,))
        return x

    @classmethod
    def new(cls):
        return cls()


class MultiClassProbabilityEndModule(Module):
    """
    The standard end module for multi classification.
    """
    def __init__(self, number_of_classes: int):
        super().__init__()
        self.number_of_classes: int = number_of_classes
        self.prediction_layer = Conv1d(in_channels=100, out_channels=self.number_of_classes, kernel_size=1)
        self.soft_max = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.prediction_layer(x)
        x = self.soft_max(x)
        x = torch.reshape(x, (-1, self.number_of_classes))
        return x

    @classmethod
    def new(cls, number_of_classes: int):
        return cls(number_of_classes)


class MultiClassScoreEndModule(Module):
    """
    The standard end module for multi classification without softmax.
    """
    def __init__(self, number_of_classes: int):
        super().__init__()
        self.number_of_classes: int = number_of_classes
        self.prediction_layer = Conv1d(in_channels=100, out_channels=self.number_of_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.prediction_layer(x)
        x = torch.reshape(x, (-1, self.number_of_classes))
        return x

    @classmethod
    def new(cls, number_of_classes: int):
        return cls(number_of_classes)
