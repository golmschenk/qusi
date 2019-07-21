"""Code for representing a dataset of lightcurves for binary classification."""
import os
import random
from typing import List

import numpy as np
import tensorflow as tf


class LightcurveDatabase:
    """A representing a dataset of lightcurves for binary classification."""
    def __init__(self, positive_data_directory: str, negative_data_directory: str,
                 positive_to_negative_data_ratio: float = None):
        self.positive_data_directory = positive_data_directory
        self.negative_data_directory = negative_data_directory
        self.time_steps_per_example = 30000
        self.batch_size = 100
        training_dataset, validation_dataset = self.generate_datasets()
        self.training_dataset: tf.data.Dataset = training_dataset
        self.validation_dataset: tf.data.Dataset = validation_dataset
        self.positive_to_negative_data_ratio = positive_to_negative_data_ratio

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """Generates the training and testing datasets."""
        positive_example_paths = [os.path.join(self.positive_data_directory, file_name) for file_name in
                                  os.listdir(self.positive_data_directory) if file_name.endswith('.npy')]
        positive_example_paths = self.remove_bad_files(positive_example_paths)
        print(f'{len(positive_example_paths)} positive examples.')
        negative_example_paths = [os.path.join(self.negative_data_directory, file_name) for file_name in
                                  os.listdir(self.negative_data_directory) if file_name.endswith('.npy')]
        negative_example_paths = self.remove_bad_files(negative_example_paths)
        print(f'{len(negative_example_paths)} negative examples.')
        positive_example_paths, negative_example_paths = self.enforce_data_ratio(positive_example_paths,
                                                                                 negative_example_paths)
        positive_labels = [1] * len(positive_example_paths)
        negative_labels = [0] * len(negative_example_paths)
        example_paths = positive_example_paths + negative_example_paths
        labels = positive_labels + negative_labels
        example_paths, labels = self.shuffle_in_unison(example_paths, labels, seed=0)
        labels = labels.astype(np.int32)
        labels = np.expand_dims(labels, axis=-1)
        file_path_dataset = tf.data.Dataset.from_tensor_slices(example_paths)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        full_dataset = tf.data.Dataset.zip((file_path_dataset, labels_dataset))
        validation_dataset_size = int(len(labels) * 0.2)
        validation_dataset = full_dataset.take(validation_dataset_size)
        training_dataset = full_dataset.skip(validation_dataset_size)
        load_and_preprocess_function = lambda file_path, label: tuple(
            tf.py_function(self.load_and_preprocess_numpy_file, [file_path, label], [tf.float32, tf.int32]))
        training_dataset = training_dataset.map(load_and_preprocess_function)
        training_dataset = training_dataset.shuffle(buffer_size=1000)
        training_dataset = training_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.map(load_and_preprocess_function)
        validation_dataset = validation_dataset.batch(self.batch_size)
        return training_dataset, validation_dataset

    def load_and_preprocess_numpy_file(self, file_path: tf.Tensor, label: int) -> (np.ndarray, int):
        """Loads numpy files from the tensor alongside labels."""
        lightcurve = np.load(file_path.numpy())
        lightcurve = self.preprocess_and_augment_lightcurve(lightcurve)
        return lightcurve.astype(np.float32), label

    def preprocess_and_augment_lightcurve(self, lightcurve: np.ndarray):
        """Slices and normalizes lightcurves."""
        lightcurve = self.remove_random_values(lightcurve)  # Helps prevent overfitting.
        lightcurve = self.roll_lightcurve(lightcurve)  # Helps prevent overfitting.
        lightcurve = self.make_uniform_length(lightcurve)  # Current network expects a fixed length.
        lightcurve = self.normalize(lightcurve)
        lightcurve = np.expand_dims(lightcurve, axis=-1)  # Network uses a "channel" dimension.
        return lightcurve

    def make_uniform_length(self, lightcurve):
        """Makes all lightcurves the same length, but clipping those too large and repeating those too small."""
        if lightcurve.shape[0] > self.time_steps_per_example:
            start_slice = np.random.randint(0, lightcurve.shape[0] - self.time_steps_per_example)
            lightcurve = lightcurve[:, :, start_slice:start_slice + self.time_steps_per_example]
        else:
            elements_to_repeat = self.time_steps_per_example - lightcurve.shape[0]
            lightcurve = np.pad(lightcurve, (0, elements_to_repeat), mode='wrap')
        return lightcurve

    def normalize(self, lightcurve):
        """Normalizes from 0 to 1 on the logarithm of the lightcurve."""
        lightcurve -= np.min(lightcurve)
        lightcurve = np.log1p(lightcurve)
        array_max = np.max(lightcurve)
        if array_max != 0:
            lightcurve /= array_max
        return lightcurve

    @staticmethod
    def remove_bad_files(file_path_list: List[str]):
        """Removes problematic lightcurves (all values the same, containing infinite or NaN values)."""
        new_file_path_list = []
        for file_path in file_path_list:
            array = np.load(file_path)
            if np.max(array) == np.min(array):
                continue
            if not np.isfinite(array).all():
                continue
            if np.isnan(array).any():
                continue
            new_file_path_list.append(file_path)
        print(f'{len(file_path_list) - len(new_file_path_list)} items with bad values removed.')
        return new_file_path_list

    @staticmethod
    def shuffle_in_unison(a, b, seed=None):
        """Shuffle two arrays in unison."""
        if seed is not None:
            np.random.seed(seed)
        indexes = np.random.permutation(len(a))
        return np.array(a)[indexes], np.array(b)[indexes]

    def remove_random_values(self, lightcurve):
        """Removes random values from the lightcurve."""
        max_values_to_remove = 10
        values_to_remove = random.randrange(max_values_to_remove)
        random_indexes = np.random.randint(0, len(lightcurve), size=values_to_remove)
        return np.delete(lightcurve, random_indexes)

    def roll_lightcurve(self, lightcurve):
        """Randomly rolls the lightcurve, moving starting elements to the end."""
        shift = np.random.randint(0, len(lightcurve))
        return np.roll(lightcurve, shift)

    def enforce_data_ratio(self, positive_examples, negative_examples):
        """Repeats examples to enforce a given training ratio."""
        existing_ratio = len(positive_examples) / len(negative_examples)
        if existing_ratio < self.positive_to_negative_data_ratio:
            desired_number_of_positive_examples = int(self.positive_to_negative_data_ratio * len(negative_examples))
            additional_positive_examples_needed = desired_number_of_positive_examples - len(positive_examples)
            positive_examples = np.pad(positive_examples, (0, additional_positive_examples_needed), mode='wrap')
        else:
            desired_number_of_negative_examples = int(
                (1 / self.positive_to_negative_data_ratio) * len(positive_examples))
            additional_negative_examples_needed = desired_number_of_negative_examples - len(negative_examples)
            negative_examples = np.pad(negative_examples, (0, additional_negative_examples_needed), mode='wrap')
        return positive_examples, negative_examples
