"""Code for running training."""
import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy
from pathlib import Path
from ramjet.models.cura import Cura
from ramjet.models.single_layer_model import SingleLayerModel
from ramjet.photometric_database.derived.moa_survey_none_single_and_binary_database import \
    MoaSurveyNoneSingleAndBinaryDatabase
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_databases import \
    TessTwoMinuteCadenceStandardAndInjectedTransitDatabase
from ramjet.photometric_database.derived.toy_database import ToyDatabase
from ramjet.trial import create_logging_metrics, create_logging_callbacks


def train():
    """Runs the training."""
    print('Starting training process...', flush=True)
    # Basic training settings.
    database = ToyDatabase()
    model = SingleLayerModel()
    trial_name = f'{type(model).__name__}'  # Add any desired run name details to this string.
    # database.batch_size = 100  # Reducing the batch size may help if you are running out of memory.
    epochs_to_run = 1000
    logs_directory = Path('logs')

    # Setup training data, metrics, and logging.
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet', wandb_project='transit')
    training_dataset, validation_dataset = database.generate_datasets()
    loss_metric = BinaryCrossentropy(name='Loss')
    metrics = create_logging_metrics()
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    # Compile and train model.
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=5000, validation_steps=500)
    print('Training done.', flush=True)


if __name__ == '__main__':
    train()
