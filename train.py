"""Code for running training."""
import datetime
import os
import tensorflow as tf
from tensorflow.python.keras import callbacks

from lightcurve_database import LightcurveDatabase
from models import SimpleLightcurveCnn


def train():
    """Runs the training."""
    # Basic training settings.
    tf.keras.backend.set_learning_phase(True)
    model = SimpleLightcurveCnn()
    database = LightcurveDatabase()
    epochs_to_run = 1000
    trial_name = 'baseline'
    logs_directory = 'logs'

    # Prepare training data and metrics.
    training_dataset, validation_dataset = database.generate_datasets('data/positive', 'data/negative')
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.losses.BinaryCrossentropy(name='Loss')
    accuracy_metric = tf.metrics.BinaryAccuracy(name='Accuracy')
    precision_metric = tf.metrics.Precision(name='Precision')
    recall_metric = tf.metrics.Recall(name='Recall')

    # Setup logging.
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = os.path.join(logs_directory, f'{trial_name} {datetime_string}')
    tensorboard_callback = callbacks.TensorBoard(log_dir=trial_directory)

    # Compile and train model.
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=[accuracy_metric, precision_metric, recall_metric])
    try:
        model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset,
                  callbacks=[tensorboard_callback])
    except KeyboardInterrupt:
        print('Interrupted. Saving model before quitting...')
    finally:
        model.save_weights(os.path.join(trial_directory, 'model.ckpt'))

        inference_dataset = database.generate_inference_dataset('data/inference')
        dataset_tensor = tf.convert_to_tensor(inference_dataset)
        predictions = model.predict(dataset_tensor)
        print(predictions)


if __name__ == '__main__':
    train()
