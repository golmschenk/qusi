import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, Softmax
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy


class CrossEntropyAlt(Module):
    @classmethod
    def new(cls):
        return cls()

    def __init__(self):
        super().__init__()
        self.cross_entropy = CrossEntropyLoss()

    def __call__(self, preds: Tensor, target: Tensor):
        target_int = target.to(torch.int64)
        cross_entropy = self.cross_entropy(preds, target_int)
        return cross_entropy


class MulticlassAUROCAlt(Module):
    @classmethod
    def new(cls, number_of_classes: int):
        return cls(number_of_classes=number_of_classes)

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.multiclass_auroc = MulticlassAUROC(num_classes=number_of_classes)
        self.softmax = Softmax()

    def __call__(self, preds: Tensor, target: Tensor):
        probabilities = self.softmax(preds)
        target_int = target.to(torch.int64)
        cross_entropy = self.multiclass_auroc(probabilities, target_int)
        return cross_entropy


class MulticlassAccuracyAlt(Module):
    @classmethod
    def new(cls, number_of_classes: int):
        return cls(number_of_classes=number_of_classes)

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.multiclass_accuracy = MulticlassAccuracy(num_classes=number_of_classes)
        self.softmax = Softmax()

    def __call__(self, preds: Tensor, target: Tensor):
        probabilities = self.softmax(preds)
        target_int = target.to(torch.int64)
        cross_entropy = self.multiclass_accuracy(probabilities, target_int)
        return cross_entropy
