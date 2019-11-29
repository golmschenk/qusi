"""Code for a base generalized database for photometric data to be subclassed."""
import math
from abc import ABC
import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf


class LightcurveDatabase(ABC):
    """A base generalized database for photometric data to be subclassed."""
    def __init__(self, data_directory='data'):
        self.data_directory = Path(data_directory)
        self.validation_ratio = 0.2
        self.batch_size = 100
        self.trial_directory = None
        self.time_steps_per_example: int

    def log_dataset_file_names(self, dataset: tf.data.Dataset, dataset_name: str):
        """Saves the names of the files used in a dataset to a CSV file in the trail directory."""
        os.makedirs(self.trial_directory, exist_ok=True)
        training_example_paths = [example.numpy().decode('utf-8') for example in list(dataset)]
        series = pd.Series(training_example_paths)
        series.to_csv(os.path.join(self.trial_directory, f'{dataset_name}.csv'), header=False, index=False)

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

    def get_ratio_enforced_dataset(self, positive_training_dataset: tf.data.Dataset,
                                   negative_training_dataset: tf.data.Dataset,
                                   positive_to_negative_data_ratio: float) -> tf.data.Dataset:
        """Generates a dataset with an enforced data ratio."""
        if positive_to_negative_data_ratio is not None:
            positive_count = len(list(positive_training_dataset))
            negative_count = len(list(negative_training_dataset))
            existing_ratio = positive_count / negative_count
            if existing_ratio < positive_to_negative_data_ratio:
                desired_number_of_positive_examples = int(positive_to_negative_data_ratio * negative_count)
                positive_training_dataset = self.repeat_dataset_to_size(positive_training_dataset,
                                                                        desired_number_of_positive_examples)
            else:
                desired_number_of_negative_examples = int((1 / positive_to_negative_data_ratio) * positive_count)
                negative_training_dataset = self.repeat_dataset_to_size(negative_training_dataset,
                                                                        desired_number_of_negative_examples)
        return positive_training_dataset.concatenate(negative_training_dataset)

    @staticmethod
    def repeat_dataset_to_size(dataset: tf.data.Dataset, size: int) -> tf.data.Dataset:
        """Repeats a dataset to make it a desired length."""
        current_size = len(list(dataset))
        times_to_repeat = math.ceil(size / current_size)
        return dataset.repeat(times_to_repeat).take(size)

    def is_positive(self, example_path):
        """
        Checks if an example contains a microlensing event or not.

        :param example_path: The path to the example to check.
        :return: Whether or not the example contains a microlensing event.
        """
        return 'positive' in example_path

    @staticmethod
    def make_uniform_length(example: np.ndarray, length: int, randomize: bool = True) -> np.ndarray:
        """Makes the example a specific length, by clipping those too large and repeating those too small."""
        if example.shape[0] == length:
            pass
        elif example.shape[0] > length:
            if randomize:
                start_slice = np.random.randint(0, example.shape[0] - length)
            else:
                start_slice = 0
            example = example[start_slice:start_slice + length]
        else:
            elements_to_repeat = length - example.shape[0]
            example = np.pad(example, (0, elements_to_repeat), mode='wrap')
        return example

    def get_training_and_validation_datasets_for_file_paths(self, example_paths: List[Union[str, Path]]) -> (
                                                            tf.data.Dataset, tf.data.Dataset):
        """Creates a TensorFlow Dataset from a list of file names and desired label for those files."""
        example_paths = [str(example_path) for example_path in example_paths]
        np.random.seed(0)
        np.random.shuffle(example_paths)
        number_of_chunks = int(1 / self.validation_ratio)  # In this example, 5 chunks.
        chunks = np.array_split(example_paths, number_of_chunks)
        validation_chunk_index = 0  # This value should be changeable to select different validation sets.
        validation_paths = chunks[validation_chunk_index]
        training_chunks = np.delete(chunks, validation_chunk_index, axis=0)
        training_paths = np.concatenate(training_chunks)
        validation_dataset = tf.data.Dataset.from_tensor_slices(validation_paths)
        training_dataset = tf.data.Dataset.from_tensor_slices(training_paths)
        return training_dataset, validation_dataset
