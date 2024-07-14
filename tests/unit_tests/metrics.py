from unittest.mock import patch

import torch
from torch.nn import MSELoss
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryAUROC

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
