"""Code for representing a dataset of lightcurves for binary classification with a single label per example."""
import os
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class MicrolensingLabelPerExampleDatabase(LightcurveDatabase):
    """A representation of a dataset of lightcurves for binary classification with a single label per example."""
    def __init__(self):
        super().__init__()
        self.time_steps_per_example = 30000

    def generate_datasets(self, positive_data_directory, negative_data_directory,
                          positive_to_negative_data_ratio: float = None) -> (tf.data.Dataset, tf.data.Dataset):
        """Generates the training and validation datasets."""
        data_format_suffixes = ('.npy', '.pkl', '.feather')
        positive_example_paths = [os.path.join(positive_data_directory, file_name) for file_name in
                                  os.listdir(positive_data_directory) if file_name.endswith(data_format_suffixes)]
        print(f'{len(positive_example_paths)} positive examples.')
        negative_example_paths = [os.path.join(negative_data_directory, file_name) for file_name in
                                  os.listdir(negative_data_directory) if file_name.endswith(data_format_suffixes)]
        print(f'{len(negative_example_paths)} negative examples.')
        positive_datasets = self.get_training_and_validation_datasets_for_file_paths(positive_example_paths)
        positive_training_dataset, positive_validation_dataset = positive_datasets
        negative_datasets = self.get_training_and_validation_datasets_for_file_paths(negative_example_paths)
        negative_training_dataset, negative_validation_dataset = negative_datasets
        training_dataset = self.get_ratio_enforced_dataset(positive_training_dataset, negative_training_dataset,
                                                           positive_to_negative_data_ratio)
        validation_dataset = positive_validation_dataset.concatenate(negative_validation_dataset)
        if self.trial_directory is not None:
            self.log_dataset_file_names(training_dataset, dataset_name='training')
            self.log_dataset_file_names(validation_dataset, dataset_name='validation')
        load_and_preprocess_function = lambda file_path: tuple(
            tf.py_function(self.load_and_preprocess_example_file, [file_path], [tf.float32, tf.int32]))
        training_dataset = training_dataset.shuffle(buffer_size=len(list(training_dataset)))
        training_dataset = training_dataset.map(load_and_preprocess_function, num_parallel_calls=16)
        training_dataset = training_dataset.map(self.set_shape_function, num_parallel_calls=16)
        training_dataset = training_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.map(load_and_preprocess_function, num_parallel_calls=16)
        validation_dataset = validation_dataset.map(self.set_shape_function, num_parallel_calls=16)
        validation_dataset = validation_dataset.batch(self.batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset

    def set_shape_function(self, lightcurve: tf.Tensor, label: tf.Tensor):
        """
        Explicitly sets the shapes of the lightcurve and label tensor, otherwise TensorFlow can't infer it.

        :param lightcurve: The lightcurve tensor.
        :param label: The label tensor.
        :return: The lightcurve and label tensor with TensorFlow inferable shapes.
        """
        lightcurve.set_shape([self.time_steps_per_example, 1])
        label.set_shape([1])
        return lightcurve, label

    def load_and_preprocess_example_file(self, file_path: tf.Tensor) -> (np.ndarray, int):
        """Loads numpy files from the tensor alongside labels."""
        file_path_string = file_path.numpy().decode('utf-8')
        if file_path_string.endswith('.npy'):
            lightcurve = np.load(file_path_string)
        elif file_path_string.endswith('.pkl'):
            lightcurve = pd.read_pickle(file_path_string)['flux'].values
        elif file_path_string.endswith('.feather'):
            lightcurve = pd.read_feather(file_path_string)['flux'].values
        else:
            raise ValueError(f'Unknown extension when loading data from {file_path_string}')
        lightcurve = self.preprocess_and_augment_lightcurve(lightcurve)
        return lightcurve.astype(np.float32), self.is_positive(file_path_string)
        
    def preprocess_and_augment_lightcurve(self, lightcurve: np.ndarray) -> np.ndarray:
        """Prepares the lightcurves for training with several preprocessing and augmenting steps."""
        lightcurve = self.remove_random_values(lightcurve)  # Helps prevent overfitting.
        lightcurve = self.roll_lightcurve(lightcurve)  # Helps prevent overfitting.
        lightcurve = self.make_uniform_length(lightcurve, self.time_steps_per_example)  # Current network expects a fixed length.
        lightcurve = self.normalize(lightcurve)
        lightcurve = np.expand_dims(lightcurve, axis=-1)  # Network uses a "channel" dimension.
        return lightcurve

    @staticmethod
    def roll_lightcurve(lightcurve: np.ndarray) -> np.ndarray:
        """Randomly rolls the lightcurve, moving starting elements to the end."""
        shift = np.random.randint(0, len(lightcurve))
        return np.roll(lightcurve, shift)

    def generate_inference_dataset(self, inference_directory: str) -> (List[str], List[np.ndarray]):
        """Generates the inference dataset."""
        example_paths = [os.path.join(inference_directory, file_name) for file_name in
                         os.listdir(inference_directory) if file_name.endswith('.npy')]
        examples = []
        for example_path in example_paths:
            lightcurve = np.load(example_path)
            lightcurve = self.preprocess_and_augment_lightcurve(lightcurve)
            examples.append(lightcurve.astype(np.float32))
        return example_paths, examples
