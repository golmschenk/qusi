from unittest.mock import Mock

import torch
from torch.nn import ModuleList, Module
from torchmetrics import MeanSquaredError

from qusi.internal.module import QusiLightningModule, MetricGroup


class MockStateBasedMetric(Mock, MeanSquaredError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MockFunctionalMetric(Mock, Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.side_effect: Module = MeanSquaredError()


def create_fake_qusi_lightning_module() -> QusiLightningModule:
    qusi_lightning_module_mock = QusiLightningModule(
        model=Mock(return_value=torch.tensor([1])), optimizer=Mock(), train_metric_group=Mock(),
        validation_metric_groups=[Mock()]
    )
    return qusi_lightning_module_mock


def create_fake_metric_group() -> MetricGroup:
    fake_metric_group = MetricGroup(loss_metric=Mock(return_value=torch.tensor(1)),
                                    state_based_logging_metrics=ModuleList([MockStateBasedMetric()]),
                                    functional_logging_metrics=ModuleList([MockFunctionalMetric()]))
    return fake_metric_group


def test_compute_loss_and_metrics_calls_passed_loss_metric():
    fake_qusi_lightning_module0 = create_fake_qusi_lightning_module()
    fake_metric_group = create_fake_metric_group()
    batch = (torch.tensor([3]), torch.tensor([4]))
    assert not fake_metric_group.loss_metric.called
    fake_qusi_lightning_module0.compute_loss_and_metrics(batch=batch, metric_group=fake_metric_group)
    assert fake_metric_group.loss_metric.called


def test_compute_loss_and_metrics_uses_correct_phase_state_metric():
    fake_qusi_lightning_module0 = create_fake_qusi_lightning_module()
    fake_metric_group = create_fake_metric_group()
    batch = (torch.tensor([3]), torch.tensor([4]))
    assert not fake_metric_group.state_based_logging_metrics[0].called
    fake_qusi_lightning_module0.compute_loss_and_metrics(batch=batch, metric_group=fake_metric_group)
    assert fake_metric_group.state_based_logging_metrics[0].called


def test_compute_loss_and_metrics_uses_correct_phase_functional_metric():
    fake_qusi_lightning_module0 = create_fake_qusi_lightning_module()
    fake_metric_group = create_fake_metric_group()
    batch = (torch.tensor([3]), torch.tensor([4]))
    assert not fake_metric_group.functional_logging_metrics[0].called
    fake_qusi_lightning_module0.compute_loss_and_metrics(batch=batch, metric_group=fake_metric_group)
    assert fake_metric_group.functional_logging_metrics[0].called
