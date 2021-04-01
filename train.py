"""Code for running training."""
import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy
from pathlib import Path
from pathos.helpers import mp as multiprocess

from ramjet.models.hades import Hades
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_databases import \
    TessTwoMinuteCadenceStandardAndInjectedTransitDatabase
from ramjet.trial import create_logging_metrics, create_logging_callbacks
from ramjet.logging.wandb_logger import WandbLogger


def train():
    """Runs the training."""
    print('Starting training process...', flush=True)
    # Basic training settings.
    trial_name = f'baseline'  # Add any desired run name details to this string.
    database = TessTwoMinuteCadenceStandardAndInjectedTransitDatabase()
    model = Hades(database.number_of_label_types)
    # database.batch_size = 100  # Reducing the batch size may help if you are running out of memory.
    epochs_to_run = 1000
    logs_directory = Path('logs')

    # Setup training data, metrics, and logging.
    wandb_logger = WandbLogger.new(logs_directory)
    database.logger = wandb_logger
    training_dataset, validation_dataset = database.generate_datasets()
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name)
    logging_callbacks += [wandb_logger.create_callback()]
    loss_metric = BinaryCrossentropy(name='Loss')
    metrics = create_logging_metrics()
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    # Compile and train model.
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.run_eagerly = True
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=5000, validation_steps=500)
    print('Training done.', flush=True)


if __name__ == '__main__':
    train()
