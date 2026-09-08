from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path

import lightning
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.nn import BCELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from qusi.internal.distributed import ensure_torchrun_environment_variables
from qusi.internal.light_curve_dataset import InterleavedDataset
from qusi.internal.logging import set_up_default_logger
from qusi.internal.module import QusiLightningModule
from qusi.internal.progress_bar import ProgressBar
from qusi.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from qusi.internal.train_logging_configuration import TrainLoggingConfiguration
from qusi.internal.train_system_configuration import TrainSystemConfiguration

logger = logging.getLogger(__name__)


def train_session(
        train_datasets: list[Dataset],
        validation_datasets: list[Dataset],
        model: Module,
        optimizer: Optimizer | None = None,
        loss_metric: Module | None = None,
        logging_metrics: list[Module] | None = None,
        *,
        hyperparameter_configuration: TrainHyperparameterConfiguration | None = None,
        system_configuration: TrainSystemConfiguration | None = None,
        logging_configuration: TrainLoggingConfiguration | None = None,
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
    if hyperparameter_configuration is None:
        hyperparameter_configuration:TrainHyperparameterConfiguration = TrainHyperparameterConfiguration.new()
    if system_configuration is None:
        system_configuration: TrainSystemConfiguration = TrainSystemConfiguration.new()
    if loss_metric is None:
        loss_metric: Module = BCELoss()
    if logging_configuration is None:
        logging_configuration: TrainLoggingConfiguration = TrainLoggingConfiguration.new()
    if logging_metrics is None:
        logging_metrics: list[Module] = [BinaryAccuracy(), BinaryAUROC()]

    ensure_torchrun_environment_variables()
    set_up_default_logger()

    if 'QUSI_SESSION_DIRECTORY' in os.environ:
        session_directory = Path(os.environ['QUSI_SESSION_DIRECTORY'])
        sessions_directory_path = session_directory.parent
        session_name = session_directory.name
    else:
        sessions_directory_path = Path(f'sessions')
        session_name = f'{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}'
    sessions_directory_path.mkdir(exist_ok=True, parents=True)
    wandb_logger = WandbLogger(save_dir=sessions_directory_path, name=session_name, version='',
                         project=logging_configuration.wandb_project, entity=logging_configuration.wandb_entity)
    wandb_logger.log_hyperparams(logging_configuration.additional_log_dictionary)
    loggers = [CSVLogger(save_dir=sessions_directory_path, name=session_name, version=''), wandb_logger]

    progress_refresh_rate = min(100, hyperparameter_configuration.train_steps_per_cycle // 10)
    trainer = lightning.Trainer(
        max_epochs=hyperparameter_configuration.cycles,
        limit_train_batches=hyperparameter_configuration.train_steps_per_cycle,
        limit_val_batches=hyperparameter_configuration.validation_steps_per_cycle,
        log_every_n_steps=0,
        accelerator=system_configuration.accelerator,
        num_nodes=int(os.environ['WORLD_SIZE']) // int(os.environ['LOCAL_WORLD_SIZE']),
        devices=int(os.environ['LOCAL_WORLD_SIZE']),
        logger=loggers,
        callbacks=[ProgressBar(refresh_rate=progress_refresh_rate)],
    )
    # TODO: Not a fan of needing to magically pass the process number to the datasets here.
    for train_dataset in train_datasets:
        train_dataset.global_rank = trainer.global_rank
        train_dataset.world_size = trainer.world_size

    train_dataset = InterleavedDataset.new(*train_datasets)
    workers_per_dataloader = system_configuration.preprocessing_processes_per_train_process

    if hyperparameter_configuration.global_batch_size is not None:
        unrounded_local_batch_size = hyperparameter_configuration.global_batch_size / trainer.world_size
        if not unrounded_local_batch_size.is_integer():
            raise UserWarning(f'The global batch size of `{hyperparameter_configuration.global_batch_size}` is not '
                              f'divisible by the world size of `{trainer.world_size}`. Rounding to determine local '
                              f'batch size.')
        hyperparameter_configuration.local_batch_size = round(unrounded_local_batch_size)
        if hyperparameter_configuration.local_batch_size == 0:
            hyperparameter_configuration.local_batch_size = 1
    elif hyperparameter_configuration.local_batch_size is not None:
        hyperparameter_configuration.global_batch_size = (hyperparameter_configuration.local_batch_size *
                                                          trainer.world_size)
    else:
        raise ValueError('The hyperparameter configuration requires that either `global_batch_size` or '
                         '`local_batch_size` be set.')

    if workers_per_dataloader == 0:
        prefetch_factor = None
        persistent_workers = False
    else:
        prefetch_factor = 10
        persistent_workers = True
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameter_configuration.local_batch_size,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        num_workers=workers_per_dataloader,
    )
    validation_dataloaders: list[DataLoader] = []
    for validation_dataset in validation_datasets:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=hyperparameter_configuration.local_batch_size,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            num_workers=workers_per_dataloader,
        )
        validation_dataloaders.append(validation_dataloader)

    lightning_model = QusiLightningModule.new(model=model, optimizer=optimizer, loss_metric=loss_metric,
                                              logging_metrics=logging_metrics)
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloaders)
