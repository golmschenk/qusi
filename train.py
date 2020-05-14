"""Code for running training."""
import datetime
import os
import tensorflow as tf
from tensorflow.python.keras import callbacks
from tensorflow_core.python.keras.losses import BinaryCrossentropy

from ramjet.models import SimpleLightcurveCnn
from ramjet.photometric_database.toi_database import ToiDatabase


def train():
    """Runs the training."""
    print('Starting training process...', flush=True)
    # Basic training settings.
    model = SimpleLightcurveCnn()
    database = ToiDatabase()
    # database.batch_size = 100  # Reducing the batch size may help if you are running out of memory.
    epochs_to_run = 1000
    trial_name = 'baseline'
    logs_directory = 'logs'

    # Setup logging.
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = os.path.join(logs_directory, f'{trial_name} {datetime_string}')
    tensorboard_callback = callbacks.TensorBoard(log_dir=trial_directory)
    database.trial_directory = trial_directory
    model_save_path = os.path.join(trial_directory, 'model.ckpt')
    model_checkpoint_callback = callbacks.ModelCheckpoint(model_save_path, save_weights_only=True)

    # Prepare training data and metrics.
    training_dataset, validation_dataset = database.generate_datasets()
    optimizer = tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.99, beta_2=0.9999)
    loss_metric = BinaryCrossentropy(name='Loss', label_smoothing=0.1)
    metrics = [tf.metrics.BinaryAccuracy(name='Accuracy'), tf.metrics.Precision(name='Precision'),
               tf.metrics.Recall(name='Recall'),
               tf.metrics.SpecificityAtSensitivity(0.9, name='Specificity_at_90_percent_sensitivity'),
               tf.metrics.SensitivityAtSpecificity(0.9, name='Sensitivity_at_90_percent_specificity')]

    # Compile and train model.
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    try:
        model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset,
                  callbacks=[tensorboard_callback, model_checkpoint_callback], steps_per_epoch=5000,
                  validation_steps=500)
    except KeyboardInterrupt:
        print('Interrupted. Saving model before quitting...', flush=True)
    finally:
        model.save_weights(model_save_path)
    print('Training done.', flush=True)


if __name__ == '__main__':
    train()
