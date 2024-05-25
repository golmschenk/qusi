from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.nn import BCELoss, Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from qusi.internal.light_curve_dataset import InterleavedDataset, LightCurveDataset
from qusi.internal.logging import set_up_default_logger, get_metric_name
from qusi.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from qusi.internal.train_logging_configuration import TrainLoggingConfiguration
from qusi.internal.wandb_liaison import wandb_commit, wandb_init, wandb_log

logger = logging.getLogger(__name__)


def train_session(
        train_datasets: list[LightCurveDataset],
        validation_datasets: list[LightCurveDataset],
        model: Module,
        optimizer: Optimizer | None = None,
        loss_function: Module | None = None,
        metric_functions: list[Module] | None = None,
        *,
        hyperparameter_configuration: TrainHyperparameterConfiguration | None = None,
        logging_configuration: TrainLoggingConfiguration | None = None,
) -> None:
    """
    Runs a training session.

    :param train_datasets: The datasets to train on.
    :param validation_datasets: The datasets to validate on.
    :param model: The model to train.
    :param loss_function: The loss function to train the model on.
    :param metric_functions: A list of metric functions to record during the training process.
    :param hyperparameter_configuration: The configuration of the hyperparameters
    :param logging_configuration: The configuration of the logging.
    """
    if hyperparameter_configuration is None:
        hyperparameter_configuration = TrainHyperparameterConfiguration.new()
    if logging_configuration is None:
        logging_configuration = TrainLoggingConfiguration.new()
    if loss_function is None:
        loss_function = BCELoss()
    if metric_functions is None:
        metric_functions = [BinaryAccuracy(), BinaryAUROC()]
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
    torch.multiprocessing.set_start_method("spawn")
    debug = False
    if debug:
        workers_per_dataloader = 0
        prefetch_factor = None
        persistent_workers = False
    else:
        workers_per_dataloader = 10
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
    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device, non_blocking=True)
    loss_function = loss_function.to(device, non_blocking=True)
    if optimizer is None:
        optimizer = AdamW(model.parameters())
    metric_functions: list[Module] = [
        metric_function.to(device, non_blocking=True)
        for metric_function in metric_functions
    ]
    for _cycle_index in range(hyperparameter_configuration.cycles):
        logger.info(f'Cycle {_cycle_index}')
        train_phase(
            dataloader=train_dataloader,
            model=model,
            loss_function=loss_function,
            metric_functions=metric_functions,
            optimizer=optimizer,
            steps=hyperparameter_configuration.train_steps_per_cycle,
            device=device,
        )
        for validation_dataloader in validation_dataloaders:
            validation_phase(
                dataloader=validation_dataloader,
                model=model,
                loss_function=loss_function,
                metric_functions=metric_functions,
                steps=hyperparameter_configuration.validation_steps_per_cycle,
                device=device,
            )
        save_model(model, suffix="latest_model", process_rank=0)
        wandb_commit(process_rank=0)


def train_phase(
        dataloader,
        model,
        loss_function,
        metric_functions: list[Module],
        optimizer,
        steps,
        device,
):
    model.train()
    total_loss = 0
    metric_totals = np.zeros(shape=[len(metric_functions)])
    for batch_index, (input_features, targets) in enumerate(dataloader):
        # Compute prediction and loss
        # TODO: The conversion to float32 probably shouldn't be here, but the default collate_fn seems to be converting
        #  to float64. Probably should override the default collate.
        targets_on_device = targets.to(torch.float32).to(device, non_blocking=True)
        input_features_on_device = input_features.to(torch.float32).to(
            device, non_blocking=True
        )
        predicted_targets = model(input_features_on_device)
        loss = loss_function(predicted_targets, targets_on_device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = (
            loss.to(device, non_blocking=True).item(),
            (batch_index + 1) * len(input_features_on_device),
        )
        total_loss += loss
        for metric_function_index, metric_function in enumerate(metric_functions):
            batch_metric_value = metric_function(
                predicted_targets.to(device, non_blocking=True), targets_on_device
            ).item()
            metric_totals[metric_function_index] += batch_metric_value
        if batch_index % 10 == 0:
            logger.info(
                f"loss: {loss:>7f}  [{current:>5d}/{steps * len(input_features_on_device):>5d}]"
            )
        if batch_index + 1 >= steps:
            break
    wandb_log("loss", total_loss / steps, process_rank=0)
    cycle_metric_values = metric_totals / steps
    for metric_function_index, metric_function in enumerate(metric_functions):
        wandb_log(
            f"{get_metric_name(metric_function)}",
            cycle_metric_values[metric_function_index],
            process_rank=0,
        )


def validation_phase(
        dataloader, model, loss_function, metric_functions: list[Module], steps, device
):
    model.eval()
    validation_loss = 0
    metric_totals = np.zeros(shape=[len(metric_functions)])

    with torch.no_grad():
        for batch, (input_features, targets) in enumerate(dataloader):
            targets_on_device = targets.to(torch.float32).to(device, non_blocking=True)
            input_features_on_device = input_features.to(torch.float32).to(
                device, non_blocking=True
            )
            predicted_targets = model(input_features_on_device)
            validation_loss += (
                loss_function(predicted_targets, targets_on_device)
                .to(device, non_blocking=True)
                .item()
            )
            for metric_function_index, metric_function in enumerate(metric_functions):
                batch_metric_value = metric_function(
                    predicted_targets.to(device, non_blocking=True), targets_on_device
                ).item()
                metric_totals[metric_function_index] += batch_metric_value
            if batch + 1 >= steps:
                break

    validation_loss /= steps
    logger.info(f"Validation Error: \nAvg loss: {validation_loss:>8f} \n")
    wandb_log("val_loss", validation_loss, process_rank=0)
    cycle_metric_values = metric_totals / steps
    for metric_function_index, metric_function in enumerate(metric_functions):
        wandb_log(
            f"val_{get_metric_name(metric_function)}",
            cycle_metric_values[metric_function_index],
            process_rank=0,
        )


def save_model(model: Module, suffix: str, process_rank: int):
    if process_rank == 0:
        model_name = wandb.run.name
        if model_name == "":
            model_name = wandb.run.id
        torch.save(model.state_dict(), Path(f"sessions/{model_name}_{suffix}.pt"))
