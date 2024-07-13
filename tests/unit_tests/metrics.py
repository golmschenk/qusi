import torch
from torch.nn import MSELoss

from qusi.internal.train_session import update_logging_metrics


def test_update_logging_metrics_for_functional_metrics():
    predicted_targets = torch.tensor([1.])
    targets = torch.tensor([3.])
    metric_totals = torch.tensor([2.])
    expected_metric_totals = torch.tensor([6.])
    metric = MSELoss()
    update_logging_metrics(predicted_targets, targets, [metric], metric_totals)
    assert metric_totals == expected_metric_totals
