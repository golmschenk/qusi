import warnings
from unittest.mock import patch

import pytest
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy

from qusi.internal.metric import MulticlassAUROCAlt, MulticlassAccuracyAlt
from qusi.internal.train_session import update_logging_metrics, log_metrics


def test_update_logging_metrics_for_functional_metrics():
    predicted_targets = torch.tensor([1.])
    targets = torch.tensor([3.])
    metric_totals = torch.tensor([2.])
    expected_metric_totals = torch.tensor([6.])
    metric = MSELoss()
    update_logging_metrics(predicted_targets, targets, [metric], metric_totals)
    assert metric_totals == expected_metric_totals


def test_update_logging_metrics_for_state_based_torchmetrics():
    metric = MeanSquaredError()
    metric(torch.tensor([1.]), torch.tensor([2.]))  # Add some already stored metric value.
    predicted_targets = torch.tensor([1.])
    targets = torch.tensor([3.])
    metric_totals = torch.tensor([0.])
    expected_computed_metric_value = torch.tensor(2.5)  # Average of 1 and 4.
    update_logging_metrics(predicted_targets, targets, [metric], metric_totals)
    computed_metric_value = metric.compute()
    assert computed_metric_value == expected_computed_metric_value


def test_log_metrics_uses_torchmetrics_compute_if_available():
    metric = BinaryAUROC()
    predicted_targets = torch.tensor([0, 0.8])
    target = torch.tensor([0, 0])
    metric(predicted_targets, target)
    predicted_targets = torch.tensor([0.4, 0.7])
    target = torch.tensor([1, 1])
    metric(predicted_targets, target)
    placeholder_metric_totals = torch.tensor([-2.])
    with patch('qusi.internal.train_session.wandb_log') as mock_wandb_log:
        log_metrics([metric], placeholder_metric_totals, 2)
        assert mock_wandb_log.call_args.args[1] == torch.tensor(0.5)


def test_multiclass_auroc_alt_with_only_one_class_in_one_update():
    number_of_classes = 3
    metric = MulticlassAUROCAlt(number_of_classes=number_of_classes)

    predictions0 = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    targets0 = torch.tensor([1, 1], dtype=torch.float32)
    metric.update(predictions0, targets0)

    predictions1 = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]])
    targets1 = torch.tensor([0, 2], dtype=torch.float32)
    metric.update(predictions1, targets1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # Ensure warning for empty class not raised.
        result = metric.compute()

    assert result.item() == pytest.approx(0.8611, rel=0.001)


def test_multiclass_auroc_alt_with_only_one_class_in_only_update():
    number_of_classes = 3
    metric = MulticlassAUROCAlt(number_of_classes=number_of_classes)

    predictions0 = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    targets0 = torch.tensor([1, 1], dtype=torch.float32)
    metric.update(predictions0, targets0)

    with pytest.warns(UserWarning): # Ensure warning for empty class not raised.
        _ = metric.compute()


def test_multiclass_accuracy_alt_update_compute():
    metric = MulticlassAccuracyAlt(number_of_classes=3)

    predictions0 = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.05, 0.9, 0.05]])
    targets0 = torch.tensor([0, 1, 1], dtype=torch.float32)
    metric.update(predictions0, targets0)

    predictions1 = torch.tensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    targets1 = torch.tensor([1, 2], dtype=torch.float32)
    metric.update(predictions1, targets1)

    accuracy = metric.compute().item()
    assert accuracy == pytest.approx(0.888888, rel=0.001)
