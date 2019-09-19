"""Code for a base generalized database for photometric data to be subclassed."""

from abc import ABC
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf


class LightcurveDatabase(ABC):
    """A base generalized database for photometric data to be subclassed."""
    def __init__(self):
        self.batch_size = 100
        self.trial_directory = None

    def log_dataset_file_names(self, dataset: tf.data.Dataset, dataset_name: str):
        """Saves the names of the files used in a dataset to a CSV file in the trail directory."""
        os.makedirs(self.trial_directory, exist_ok=True)
        training_example_paths = [example[0].numpy().decode('utf-8') for example in list(dataset)]
        series = pd.Series(training_example_paths)
        series.to_csv(os.path.join(self.trial_directory, f'{dataset_name}.csv'), header=False, index=False)

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

    @staticmethod
    def normalize(lightcurve: np.ndarray) -> np.ndarray:
        """Normalizes from 0 to 1 on the logarithm of the lightcurve."""
        lightcurve -= np.min(lightcurve)
        lightcurve = np.log1p(lightcurve)
        array_max = np.max(lightcurve)
        if array_max != 0:
            lightcurve /= array_max
        return lightcurve

    @staticmethod
    def shuffle_in_unison(a, b, seed=None):
        """Shuffle two arrays in unison."""
        if seed is not None:
            np.random.seed(seed)
        indexes = np.random.permutation(len(a))
        return np.array(a)[indexes], np.array(b)[indexes]

    @staticmethod
    def remove_random_values(lightcurve: np.ndarray) -> np.ndarray:
        """Removes random values from the lightcurve."""
        max_values_to_remove = 10
        values_to_remove = random.randrange(max_values_to_remove)
        random_indexes = np.random.randint(0, len(lightcurve), size=values_to_remove)
        return np.delete(lightcurve, random_indexes)
