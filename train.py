"""Code for running training."""
import datetime
import os
import tensorflow as tf

from cube_database import CubeDatabase
from models import SimpleCubeCnn


def train():
    """Runs the training."""
    # Setup the trial parameters.
    model = SimpleCubeCnn()
    database = CubeDatabase('data/positive', 'data/negative')
    training_dataset = database.training_dataset
    validation_dataset = database.validation_dataset
    epochs_to_run = 1000
    trial_name = 'baseline'
    logs_directory = 'logs'

    # Setup the training and evaluation metrics.
    loss_function = tf.losses.BinaryCrossentropy()
    optimizer = tf.optimizers.Adam()
    training_loss_metric = tf.metrics.Mean(name='Training Loss')
    training_accuracy_metric = tf.metrics.BinaryAccuracy(name='Training Accuracy')
    validation_loss_metric = tf.metrics.Mean(name='Validation Loss')
    validation_accuracy_metric = tf.metrics.BinaryAccuracy(name='Validation Accuracy')

    # Define the training and evaluation functions.
    @tf.function
    def train_step(training_examples, training_labels):
        """Runs the training step."""
        with tf.GradientTape() as tape:
            predictions = model(training_examples, training=True)
            predictions = tf.reshape(predictions, [-1])
            training_loss = loss_function(training_labels, predictions)
        gradients = tape.gradient(training_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        training_loss_metric(training_loss)
        training_accuracy_metric(training_labels, predictions)

    @tf.function
    def validation_step(validation_examples, validation_labels):
        """Runs the testing step."""
        predictions = model(validation_examples, training=False)
        predictions = tf.reshape(predictions, [-1])
        validation_loss = loss_function(validation_labels, predictions)
        validation_loss_metric(validation_loss)
        validation_accuracy_metric(validation_labels, predictions)

    # Prepare the logging.
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = os.path.join(logs_directory, f'{trial_name} {datetime_string}')
    training_log_directory = os.path.join(trial_directory, 'train')
    validation_log_directory = os.path.join(trial_directory, 'validation')
    train_summary_writer = tf.summary.create_file_writer(training_log_directory)
    test_summary_writer = tf.summary.create_file_writer(validation_log_directory)

    # Run the training.
    for epoch in range(epochs_to_run):
        for training_images_, training_labels_ in training_dataset:
            train_step(training_images_, training_labels_)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', training_loss_metric.result(), step=epoch)
            tf.summary.scalar('accuracy', training_accuracy_metric.result(), step=epoch)
        for validation_images_, validation_labels_ in validation_dataset:
            validation_step(validation_images_, validation_labels_)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', validation_loss_metric.result(), step=epoch)
            tf.summary.scalar('accuracy', validation_accuracy_metric.result(), step=epoch)
        print(f'Epoch {epoch}, '
              f'Training Loss: {training_loss_metric.result():.6f}, '
              f'Training Accuracy: {training_accuracy_metric.result() * 100:.4f}, '
              f'Validation Loss: {validation_loss_metric.result():.6f}, '
              f'Validation Accuracy: {validation_accuracy_metric.result() * 100:.4f}')
        # Reset metrics every epoch
        training_loss_metric.reset_states()
        training_accuracy_metric.reset_states()
        validation_loss_metric.reset_states()
        validation_accuracy_metric.reset_states()


if __name__ == '__main__':
    train()
