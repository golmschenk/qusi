import os
import tempfile
from pathlib import Path

import tensorflow as tf
import pytest
from tensorflow.keras.losses import BinaryCrossentropy

from ramjet.models.single_layer_model import SingleLayerModel, SingleLayerModelWithAuxiliary
from ramjet.photometric_database.derived.toy_database import ToyRamjetDatabase, ToyRamjetDatabaseWithAuxiliary
from ramjet.trial import create_logging_callbacks, create_logging_metrics


@pytest.mark.integration
@pytest.mark.slow
def test_ramjet_train():
    """Tests a complete training setup with a toy database and model."""
    os.environ['WANDB_API_KEY'] = 'fake_key'
    os.environ['WANDB_MODE'] = 'offline'
    database = ToyRamjetDatabase()
    model = SingleLayerModel()
    trial_name = f'{type(model).__name__}'
    epochs_to_run = 2
    logs_directory = Path(tempfile.gettempdir()).joinpath('train')
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet', wandb_project='transit')
    training_dataset, validation_dataset = database.generate_datasets()
    loss_metric = BinaryCrossentropy(name='Loss')
    metrics = create_logging_metrics()
    optimizer = tf.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.run_eagerly = True
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=2, validation_steps=2)


@pytest.mark.integration
@pytest.mark.slow
def test_train_with_auxiliary_information():
    """Tests a complete training setup with a toy database and model using auxiliary information."""
    os.environ['WANDB_API_KEY'] = 'fake_key'
    os.environ['WANDB_MODE'] = 'offline'
    database = ToyRamjetDatabaseWithAuxiliary()
    model = SingleLayerModelWithAuxiliary()
    trial_name = f'{type(model).__name__}'
    epochs_to_run = 2
    logs_directory = Path(tempfile.gettempdir()).joinpath('train')
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet', wandb_project='transit')
    training_dataset, validation_dataset = database.generate_datasets()
    loss_metric = BinaryCrossentropy(name='Loss')
    metrics = create_logging_metrics()
    optimizer = tf.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.run_eagerly = True
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=2, validation_steps=2)
