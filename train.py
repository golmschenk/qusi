"""Code for running training."""
import datetime
import os
import tensorflow as tf
from tensorflow.python.keras import callbacks

from photometric_database.microlensing_label_per_example_database import MicrolensingLabelPerExampleDatabase
from models import SimpleLightcurveCnn


def train():
    """Runs the training."""
    # Basic training settings.
    model = SimpleLightcurveCnn()
    database = MicrolensingLabelPerExampleDatabase()
    epochs_to_run = 1000
    trial_name = 'baseline'
    logs_directory = 'logs'

    # Setup logging.
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = os.path.join(logs_directory, f'{trial_name} {datetime_string}')
    tensorboard_callback = callbacks.TensorBoard(log_dir=trial_directory)
    database.trial_directory = trial_directory

    # Prepare training data and metrics.
    training_dataset, validation_dataset = database.generate_datasets('data/positive', 'data/negative',
                                                                      positive_to_negative_data_ratio=1)
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.losses.BinaryCrossentropy(name='Loss')
    metrics = [tf.metrics.BinaryAccuracy(name='Accuracy'), tf.metrics.Precision(name='Precision'),
               tf.metrics.Recall(name='Recall'),
               tf.metrics.SpecificityAtSensitivity(0.9, name='Specificity_at_90_percent_sensitivity'),
               tf.metrics.SensitivityAtSpecificity(0.9, name='Sensitivity_at_90_percent_specificity')]

    # Compile and train model.
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    try:
        model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset,
                  callbacks=[tensorboard_callback])
    except KeyboardInterrupt:
        print('Interrupted. Saving model before quitting...')
    finally:
        model.save_weights(os.path.join(trial_directory, 'model.ckpt'))


if __name__ == '__main__':
    train()
