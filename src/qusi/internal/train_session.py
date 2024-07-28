from __future__ import annotations

import logging
from pathlib import Path
from warnings import warn

import numpy as np
import torch
import wandb
from torch.nn import BCELoss, Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from qusi.internal.light_curve_dataset import InterleavedDataset, LightCurveDataset
from qusi.internal.logging import set_up_default_logger, get_metric_name
from qusi.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from qusi.internal.train_logging_configuration import TrainLoggingConfiguration
from qusi.internal.train_system_configuration import TrainSystemConfiguration
from qusi.internal.wandb_liaison import wandb_commit, wandb_init, wandb_log

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
    if logging_configuration is None:
        logging_configuration = TrainLoggingConfiguration.new()
    if loss_metric is None:
        loss_metric = BCELoss()
    if logging_metrics is None:
        logging_metrics = [BinaryAccuracy(), BinaryAUROC()]

    set_up_default_logger()
    sessions_directory = Path("sessions")
    sessions_directory.mkdir(exist_ok=True)
    wandb_init(
        process_rank=0,
        project=logging_configuration.wandb_project,
        entity=logging_configuration.wandb_entity,
        settings=wandb.Settings(start_method="thread"),
        dir=sessions_directory,
    )
    train_dataset = InterleavedDataset.new(*train_datasets)
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device, non_blocking=True)
    loss_metric = loss_metric.to(device, non_blocking=True)
    if optimizer is None:
        optimizer = AdamW(model.parameters())
    logging_metrics: list[Module] = [
        metric_function.to(device, non_blocking=True)
        for metric_function in logging_metrics
    ]
    for _cycle_index in range(hyperparameter_configuration.cycles):
        logger.info(f'Cycle {_cycle_index}')
        train_phase(dataloader=train_dataloader, model=model, loss_metric=loss_metric,
                    logging_metrics=logging_metrics, optimizer=optimizer,
                    steps=hyperparameter_configuration.train_steps_per_cycle, device=device)
        for validation_dataloader in validation_dataloaders:
            validation_phase(dataloader=validation_dataloader, model=model, loss_metric=loss_metric,
                             logging_metrics=logging_metrics,
                             steps=hyperparameter_configuration.validation_steps_per_cycle, device=device)
        save_model(model, suffix="latest_model", process_rank=0)
        wandb_commit(process_rank=0)


def train_phase(
        dataloader,
        model,
        loss_metric,
        logging_metrics: list[Module],
        optimizer,
        steps,
        device,
):
    model.train()
    total_loss = 0
    metric_totals = np.zeros(shape=[len(logging_metrics)])
    for batch_index, (input_features, targets) in enumerate(dataloader):
        # Compute prediction and loss
        # TODO: The conversion to float32 probably shouldn't be here, but the default collate_fn seems to be converting
        #  to float64. Probably should override the default collate.
        targets_on_device = targets.to(torch.float32).to(device, non_blocking=True)
        input_features_on_device = input_features.to(torch.float32).to(
            device, non_blocking=True
        )
        predicted_targets = model(input_features_on_device)
        loss = loss_metric(predicted_targets, targets_on_device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = (
            loss.to(device, non_blocking=True).item(),
            (batch_index + 1) * len(input_features_on_device),
        )
        total_loss += loss
        update_logging_metrics(predicted_targets, targets_on_device, logging_metrics, metric_totals)
        if batch_index % 10 == 0:
            logger.info(
                f"loss: {loss:>7f}  [{current:>5d}/{steps * len(input_features_on_device):>5d}]"
            )
        if batch_index + 1 >= steps:
            break
    wandb_log("loss", total_loss / steps, process_rank=0)
    log_metrics(logging_metrics, metric_totals, steps)


def update_logging_metrics(predicted_targets, targets_on_device, logging_metrics, metric_totals):
    for logging_metric_index, logging_metric in enumerate(logging_metrics):
        batch_metric_value = logging_metric(
            predicted_targets, targets_on_device
        ).item()
        metric_totals[logging_metric_index] += batch_metric_value


def validation_phase(
        dataloader,
        model,
        loss_metric,
        logging_metrics: list[Module],
        steps,
        device
):
    model.eval()
    validation_loss = 0
    metric_totals = np.zeros(shape=[len(logging_metrics)])

    with torch.no_grad():
        for batch, (input_features, targets) in enumerate(dataloader):
            targets_on_device = targets.to(torch.float32).to(device, non_blocking=True)
            input_features_on_device = input_features.to(torch.float32).to(
                device, non_blocking=True
            )
            predicted_targets = model(input_features_on_device)
            validation_loss += (
                loss_metric(predicted_targets, targets_on_device)
                .to(device, non_blocking=True)
                .item()
            )
            update_logging_metrics(predicted_targets, targets_on_device, logging_metrics, metric_totals)
            if batch + 1 >= steps:
                break

    validation_loss /= steps
    logger.info(f"Validation Error: \nAvg loss: {validation_loss:>8f} \n")
    wandb_log("val_loss", validation_loss, process_rank=0)
    log_prefix = 'val_'
    log_metrics(logging_metrics, metric_totals, steps, log_prefix)


def log_metrics(logging_metrics, metric_totals, steps, log_prefix: str = ''):
    cycle_metric_values = metric_totals / steps
    for logging_metric_index, logging_metric in enumerate(logging_metrics):
        if isinstance(logging_metric, Metric):
            metric_value = logging_metric.compute()
        else:
            metric_value = cycle_metric_values[logging_metric_index]
        wandb_log(
            f'{log_prefix}{get_metric_name(logging_metric)}',
            metric_value,
            process_rank=0,
        )


def save_model(model: Module, suffix: str, process_rank: int):
    if process_rank == 0:
        model_name = wandb.run.name
        if model_name == "":
            model_name = wandb.run.id
        torch.save(model.state_dict(), Path(f"sessions/{model_name}_{suffix}.pt"))
