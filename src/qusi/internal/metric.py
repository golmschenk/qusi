import torch
from torch import Tensor
from torch.nn import NLLLoss, Module, CrossEntropyLoss
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy


class CrossEntropyAlt(Module):
    @classmethod
    def new(cls):
        return cls()

    def __init__(self):
        super().__init__()
        self.nll_loss = NLLLoss()

    def __call__(self, preds: Tensor, target: Tensor):
        predicted_log_probabilities = torch.log(preds)
        target_int = target.to(torch.int64)
        cross_entropy = self.nll_loss(predicted_log_probabilities, target_int)
        return cross_entropy

class CrossEntropyAlt2(Module):  # TODO: Temporary test for Abhina.
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

    def __call__(self, preds: Tensor, target: Tensor):
        target_int = target.to(torch.int64)
        cross_entropy = self.multiclass_auroc(preds, target_int)
        return cross_entropy

class MulticlassAccuracyAlt(Module):
    @classmethod
    def new(cls, number_of_classes: int):
        return cls(number_of_classes=number_of_classes)

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.multiclass_accuracy = MulticlassAccuracy(num_classes=number_of_classes)

    def __call__(self, preds: Tensor, target: Tensor):
        target_int = target.to(torch.int64)
        cross_entropy = self.multiclass_accuracy(preds, target_int)
        return cross_entropy
