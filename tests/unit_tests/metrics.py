import torch
from torch.nn import MSELoss
from torchmetrics import MeanSquaredError

from qusi.internal.train_session import update_logging_metrics


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
    value = metric(torch.tensor([1.]), torch.tensor([2.]))  # Add some already stored metric value.
    predicted_targets = torch.tensor([1.])
    targets = torch.tensor([3.])
    metric_totals = torch.tensor([0.])
    expected_computed_metric_value = torch.tensor(2.5)  # Average of 1 and 4.
    update_logging_metrics(predicted_targets, targets, [metric], metric_totals)
    computed_metric_value = metric.compute()
    assert computed_metric_value == expected_computed_metric_value
