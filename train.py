#Tells the script not to use the GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""Code for running training."""
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from pathlib import Path
from ramjet.models.cura import Cura
from ramjet.models.hades import Hades, HadesWithoutBatchNormalization
from ramjet.photometric_database.derived.moa_survey_microlensing_and_non_microlening_database import \
    MoaSurveyMicrolensingAndNonMicrolensingWithHardCasesDatabase, MoaSurveyMicrolensingAndNonMicrolensingDatabase, \
    MoaSurveyMicrolensingAndNonMicrolensingWithTimeDatabase, \
    MoaSurveyMicrolensingAndNonMicrolensingWithTimeAndHardCasesDatabase
from ramjet.trial import create_logging_metrics, create_logging_callbacks


def train(test_split):
    """Runs the training."""
    print('Starting training process...', flush=True)
    # Basic training settings.
    database = MoaSurveyMicrolensingAndNonMicrolensingDatabase(test_split=test_split)
    model = Hades(database.number_of_label_values)
    # model = Cura(database.number_of_label_values, database.number_of_input_channels)
    # database.batch_size = 50  # Reducing the batch size may help if you are running out of memory.
    # trial_name = f'{type(model).__name__}_TIME_Hard_test_split_{test_split}'  # Add any desired run name details to this string.
    trial_name = f'{type(model).__name__}_test_split_{test_split}'  # Add any desired run name details to this string.
    epochs_to_run = 30
    logs_directory = Path('logs')

    # Setup training data, metrics, and logging.
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet',
                                                 wandb_project='microlensing_non_microlensing_improvements')
    training_dataset, validation_dataset = database.generate_datasets()
    loss_metric = BinaryCrossentropy(name='Loss')
    metrics = create_logging_metrics()
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    # optimizer = tf.optimizers.Nadam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

    # Compile and train model.
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=5000, validation_steps=500)
    print('Training done.', flush=True)
    # `database` disappears


if __name__ == '__main__':
    train(test_split=1)
