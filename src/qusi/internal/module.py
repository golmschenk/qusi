import copy
from typing import Any

import numpy as np
import numpy.typing as npt
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
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
        self.loss_cycle_total: float = 0
        self.steps_run_in_phase: int = 0
        self.functional_logging_metric_cycle_totals: npt.NDArray = np.zeros(
            len(self.functional_logging_metrics), dtype=np.float32)

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
            optimizer: Optimizer | None,
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
                       validation_metric_groups=[validation_metric_group])
        return instance

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            train_metric_group: MetricGroup,
            validation_metric_groups: list[MetricGroup],
    ):
        super().__init__()
        self.model: Module = model
        self._optimizer: Optimizer = optimizer
        self.train_metric_group = train_metric_group
        self.validation_metric_groups: list[MetricGroup] = validation_metric_groups

    def forward(self, inputs: Any) -> Any:
        return self.model(inputs)

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
        self.log_loss_and_metrics()

    def log_loss_and_metrics(self, logging_name_prefix: str = ''):
        for state_based_logging_metric in self.train_state_based_logging_metrics:
            state_based_logging_metric_name = get_metric_name(state_based_logging_metric)
            self.log(name=logging_name_prefix + state_based_logging_metric_name,
                     value=state_based_logging_metric.compute(), sync_dist=True)
            state_based_logging_metric.reset()
        for functional_logging_metric_index, functional_logging_metric in enumerate(
                self.train_functional_logging_metrics):
            functional_logging_metric_name = get_metric_name(functional_logging_metric)
            functional_logging_metric_cycle_total = float(self._train_functional_logging_metric_cycle_totals[
                                                              functional_logging_metric_index])

            functional_logging_metric_cycle_mean = functional_logging_metric_cycle_total / self._train_steps_run_in_phase
            self.log(name=logging_name_prefix + functional_logging_metric_name,
                     value=functional_logging_metric_cycle_mean,
                     sync_dist=True)
        mean_cycle_loss = self._train_loss_cycle_total / self._train_steps_run_in_phase
        self.log(name=logging_name_prefix + 'loss',
                 value=mean_cycle_loss, sync_dist=True)
        self._train_loss_cycle_total = 0
        self._train_functional_logging_metric_cycle_totals = np.zeros(len(self.train_functional_logging_metrics),
                                                                      dtype=np.float32)
        self._train_steps_run_in_phase = 0

    def validation_step(self, batch: tuple[Any, Any], batch_index: int) -> STEP_OUTPUT:
        return self.compute_loss_and_metrics(batch, self.validation_metric_groups[0])

    def on_validation_epoch_end(self) -> None:
        self.log_loss_and_metrics(logging_name_prefix='val_')

    def configure_optimizers(self):
        return self._optimizer
