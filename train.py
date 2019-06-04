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
    loss_function = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    training_loss_metric = tf.keras.metrics.Mean(name='Training Loss')
    training_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='Training Accuracy')
    validation_loss_metric = tf.keras.metrics.Mean(name='Validation Loss')
    validation_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='Validation Accuracy')

    # Define the training and evaluation functions.
    @tf.function(autograph=False)
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

    @tf.function(autograph=False)
    def validation_step(testing_examples, testing_labels):
        """Runs the testing step."""
        predictions = model(testing_examples, training=False)
        predictions = tf.reshape(predictions, [-1])
        validation_loss = loss_function(testing_labels, predictions)
        validation_loss_metric(validation_loss)
        validation_accuracy_metric(testing_labels, predictions)

    # Prepare the logging.
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = os.path.join(logs_directory, f'{trial_name} {datetime_string}')
    train_log_dir = os.path.join(trial_directory, 'train')
    test_log_dir = os.path.join(trial_directory, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Run the training.
    for epoch in range(epochs_to_run):
        for images, labels in training_dataset:
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', training_loss_metric.result(), step=epoch)
            tf.summary.scalar('accuracy', training_accuracy_metric.result(), step=epoch)
        for validation_images, validation_labels in validation_dataset:
            validation_step(validation_images, validation_labels)
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
