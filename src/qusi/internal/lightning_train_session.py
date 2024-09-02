from __future__ import annotations

import logging
from warnings import warn

import lightning
from torch.nn import BCELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from qusi.internal.light_curve_dataset import InterleavedDataset, LightCurveDataset
from qusi.internal.logging import set_up_default_logger
from qusi.internal.module import QusiLightningModule
from qusi.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from qusi.internal.train_logging_configuration import TrainLoggingConfiguration
from qusi.internal.train_system_configuration import TrainSystemConfiguration

logger = logging.getLogger(__name__)


def train_session(
        train_datasets: list[LightCurveDataset],
        validation_datasets: list[LightCurveDataset],
        model: Module,
        optimizer: Optimizer | None = None,
        loss_metric: Module | None = None,
        logging_metrics: list[Module] | None = None,
        *,
        hyperparameter_configuration: TrainHyperparameterConfiguration | None = None,
        system_configuration: TrainSystemConfiguration | None = None,
        logging_configuration: TrainLoggingConfiguration | None = None,
        # Deprecated keyword parameters.
        loss_function: Module | None = None,
        metric_functions: list[Module] | None = None,
) -> None:
    """
    Runs a training session.

    :param train_datasets: The datasets to train on.
    :param validation_datasets: The datasets to validate on.
    :param model: The model to train.
    :param optimizer: The optimizer to be used during training.
    :param loss_metric: The loss function to train the model on.
    :param logging_metrics: A list of metric functions to record during the training process.
    :param hyperparameter_configuration: The configuration of the hyperparameters.
    :param system_configuration: The configuration of the system.
    :param logging_configuration: The configuration of the logging.
    """
    if loss_metric is not None and loss_function is not None:
        raise ValueError('Both `loss_metric` and `loss_function` cannot be set at the same time.')
    if logging_metrics is not None and metric_functions is not None:
        raise ValueError('Both `logging_metrics` and `metric_functions` cannot be set at the same time.')
    if loss_function is not None:
        warn('`loss_function` is deprecated and will be removed in the future. '
             'Please use `loss_metric` instead.', UserWarning)
        loss_metric = loss_function
    if metric_functions is not None:
        warn('`metric_functions` is deprecated and will be removed in the future. '
             'Please use `logging_metrics` instead.', UserWarning)
        logging_metrics = metric_functions

    if hyperparameter_configuration is None:
        hyperparameter_configuration = TrainHyperparameterConfiguration.new()
    if system_configuration is None:
        system_configuration = TrainSystemConfiguration.new()
    if loss_metric is None:
        loss_metric = BCELoss()
    if logging_metrics is None:
        logging_metrics = [BinaryAccuracy(), BinaryAUROC()]

    set_up_default_logger()
    train_dataset = InterleavedDataset.new(*train_datasets)
    workers_per_dataloader = system_configuration.preprocessing_processes_per_train_process
    if workers_per_dataloader == 0:
        prefetch_factor = None
        persistent_workers = False
    else:
        prefetch_factor = 10
        persistent_workers = True
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameter_configuration.batch_size,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        num_workers=workers_per_dataloader,
    )
    validation_dataloaders: list[DataLoader] = []
    for validation_dataset in validation_datasets:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=hyperparameter_configuration.batch_size,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            num_workers=workers_per_dataloader,
        )
        validation_dataloaders.append(validation_dataloader)

    lightning_model = QusiLightningModule.new(model=model, optimizer=optimizer, loss_metric=loss_metric,
                                              logging_metrics=logging_metrics)
    trainer = lightning.Trainer(
        max_epochs=hyperparameter_configuration.cycles,
        limit_train_batches=hyperparameter_configuration.train_steps_per_cycle,
        limit_val_batches=hyperparameter_configuration.validation_steps_per_cycle,
    )
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloaders)
