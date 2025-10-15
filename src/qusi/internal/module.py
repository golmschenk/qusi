from __future__ import annotations

import copy
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, tensor
from torch.nn import Module, BCELoss, ModuleList
from torch.optim import Optimizer, AdamW
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from typing_extensions import Self

from qusi.internal.logging import get_metric_name


class MetricGroup(Module):
    def __init__(self, loss_metric: Module, state_based_logging_metrics: ModuleList,
                 functional_logging_metrics: ModuleList):
        super().__init__()
        self.loss_metric: Module = loss_metric
        self.state_based_logging_metrics: ModuleList = state_based_logging_metrics
        self.functional_logging_metrics: ModuleList = functional_logging_metrics
        # Lightning requires tensors be registered to be automatically moved between devices.
        # Then we assign it to itself to force IDE resolution.
        self.register_buffer('loss_cycle_total', tensor(0, dtype=torch.float32))
        self.loss_cycle_total: Tensor = self.loss_cycle_total
        self.register_buffer('steps_run_in_phase', tensor(0, dtype=torch.int64))
        self.steps_run_in_phase: Tensor = self.steps_run_in_phase
        self.register_buffer('functional_logging_metric_cycle_totals',
                             torch.zeros(len(self.functional_logging_metrics), dtype=torch.float32))
        self.functional_logging_metric_cycle_totals: Tensor = self.functional_logging_metric_cycle_totals

    @classmethod
    def new(
            cls,
            loss_metric: Module,
            state_based_logging_metrics: ModuleList,
            functional_logging_metrics: ModuleList
    ) -> Self:
        loss_metric_: Module = copy.deepcopy(loss_metric)
        state_based_logging_metrics_: ModuleList = copy.deepcopy(state_based_logging_metrics)
        functional_logging_metrics_: ModuleList = copy.deepcopy(functional_logging_metrics)
        instance = cls(loss_metric=loss_metric_, state_based_logging_metrics=state_based_logging_metrics_,
                       functional_logging_metrics=functional_logging_metrics_)
        return instance


class QusiLightningModule(LightningModule):
    @classmethod
    def new(
            cls,
            model: Module,
            optimizer: Optimizer | None = None,
            loss_metric: Module | None = None,
            logging_metrics: list[Module] | None = None,
    ) -> Self:
        if optimizer is None:
            optimizer = AdamW(model.parameters())
        if loss_metric is None:
            loss_metric = BCELoss()
        if logging_metrics is None:
            logging_metrics = [BinaryAccuracy(), BinaryAUROC()]
        state_based_logging_metrics: ModuleList = ModuleList()
        functional_logging_metrics: ModuleList = ModuleList()
        for logging_metric in logging_metrics:
            if isinstance(logging_metric, Metric):
                state_based_logging_metrics.append(logging_metric)
            else:
                functional_logging_metrics.append(logging_metric)
        train_metric_group = MetricGroup.new(loss_metric, state_based_logging_metrics, functional_logging_metrics)
        validation_metric_group = MetricGroup.new(loss_metric, state_based_logging_metrics, functional_logging_metrics)
        instance = cls(model=model, optimizer=optimizer, train_metric_group=train_metric_group,
                       validation_metric_groups=ModuleList([validation_metric_group]))
        return instance

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            train_metric_group: MetricGroup,
            validation_metric_groups: ModuleList,
    ):
        super().__init__()
        self.model: Module = model
        self._optimizer: Optimizer = optimizer
        self.train_metric_group: MetricGroup = train_metric_group
        self.validation_metric_groups: ModuleList | list[MetricGroup] = validation_metric_groups
        # Lightning requires tensors are registered to be automatically moved between devices.
        # Then we assign it to itself to force IDE resolution.
        # `cycle` is incremented and logged during the train epoch start, so it needs to start at -1.
        self.register_buffer('cycle', tensor(-1, dtype=torch.int64))
        self.cycle: Tensor = self.cycle

    def forward(self, inputs: Any) -> Any:
        return self.model(inputs)

    def on_train_epoch_start(self) -> None:
        # Due to Lightning's inconsistent step ordering, performing this during the train epoch start gives the most
        # consistent results.
        self.cycle += 1

    def training_step(self, batch: tuple[Any, Any], batch_index: int) -> STEP_OUTPUT:
        return self.compute_loss_and_metrics(batch, self.train_metric_group)

    def compute_loss_and_metrics(self, batch: tuple[Any, Any], metric_group: MetricGroup):
        inputs, target = batch
        predicted = self(inputs)
        loss = metric_group.loss_metric(predicted, target)
        metric_group.loss_cycle_total += loss
        for state_based_logging_metric in metric_group.state_based_logging_metrics:
            state_based_logging_metric(predicted, target)
        for functional_logging_metric_index, functional_logging_metric in enumerate(
                metric_group.functional_logging_metrics):
            functional_logging_metric_value = functional_logging_metric(predicted, target)
            metric_group.functional_logging_metric_cycle_totals[
                functional_logging_metric_index] += functional_logging_metric_value
        metric_group.steps_run_in_phase += 1
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_loss_and_metrics(self.train_metric_group, logging_name_prefix='')

    def log_loss_and_metrics(self, metric_group: MetricGroup, logging_name_prefix: str = ''):
        mean_cycle_loss = metric_group.loss_cycle_total / metric_group.steps_run_in_phase
        self.log(name=logging_name_prefix + 'loss',
                 value=mean_cycle_loss, sync_dist=True, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log(name='cycle', value=self.cycle, reduce_fx=torch.max, rank_zero_only=True, on_step=False, on_epoch=True)
        for state_based_logging_metric in metric_group.state_based_logging_metrics:
            state_based_logging_metric_name = get_metric_name(state_based_logging_metric)
            self.log(name=logging_name_prefix + state_based_logging_metric_name,
                     value=state_based_logging_metric.compute(), sync_dist=True, on_step=False, on_epoch=True)
            state_based_logging_metric.reset()
        for functional_logging_metric_index, functional_logging_metric in enumerate(
                metric_group.functional_logging_metrics):
            functional_logging_metric_name = get_metric_name(functional_logging_metric)
            functional_logging_metric_cycle_total = float(metric_group.functional_logging_metric_cycle_totals[
                                                              functional_logging_metric_index])

            functional_logging_metric_cycle_mean = functional_logging_metric_cycle_total / metric_group.steps_run_in_phase
            self.log(name=logging_name_prefix + functional_logging_metric_name,
                     value=functional_logging_metric_cycle_mean,
                     sync_dist=True, on_step=False, on_epoch=True)
        metric_group.loss_cycle_total.zero_()
        metric_group.steps_run_in_phase.zero_()
        metric_group.functional_logging_metric_cycle_totals.zero_()

    def validation_step(self, batch: tuple[Any, Any], batch_index: int) -> STEP_OUTPUT:
        return self.compute_loss_and_metrics(batch, self.validation_metric_groups[0])

    def on_validation_epoch_end(self) -> None:
        self.log_loss_and_metrics(self.validation_metric_groups[0], logging_name_prefix='val_')

    def configure_optimizers(self):
        return self._optimizer
