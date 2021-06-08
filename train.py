"""Code for running training."""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import wandb
from tensorflow.python.keras.losses import BinaryCrossentropy
from pathlib import Path
# from pathos.helpers import mp as multiprocess
# multiprocess.set_start_method('spawn', force=True)
from ramjet.configuration import Configuration
from ramjet.models.cura import Cura, CuraWithDropout, CuraFinalAveragePool, CuraFinalAveragePoolNarrowerer
from ramjet.models.gml_model import GmlModel, GmlModel2, GmlModel2Wider, GmlModel2LessBatchNorm, GmlModel2NoL2, \
    GmlModel2WiderNoL2, GmlModel2Wider4NoL2, GmlModel2Wider4NoL2NoDo, GmlModel2Wider4, GmlModel3, GmlModel3Narrower, \
    GmlModel3NarrowerNoL2
from ramjet.models.hades import Hades
from ramjet.photometric_database.derived.moa_survey_none_single_and_binary_database import \
    MoaSurveyNoneSingleAndBinaryDatabase
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_databases import \
    TessTwoMinuteCadenceStandardAndInjectedTransitDatabase
from ramjet.trial import create_logging_metrics, create_logging_callbacks
from ramjet.logging.wandb_logger import WandbLogger


def train():
    """Runs the training."""
    print('Starting training process...', flush=True)
    # Basic training settings.
    database = MoaSurveyNoneSingleAndBinaryDatabase()
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
    # if True:
        model = CuraFinalAveragePoolNarrowerer(database.number_of_label_types, database.number_of_input_channels)
        trial_name = f'{type(model).__name__}'  # Add any desired run name details to this string.
        # database.batch_size = 100  # Reducing the batch size may help if you are running out of memory.
        epochs_to_run = 1000
        logs_directory = Path('logs')

        # Setup training data, metrics, and logging.
        logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database)
        training_dataset, validation_dataset = database.generate_datasets()
        wandb.run.notes = trial_name
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
