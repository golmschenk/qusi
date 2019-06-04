"""Code for representing a dataset of TESS data cubes for binary classification."""
import os
import random
from typing import List

import numpy as np
import tensorflow as tf


class CubeDatabase:
    """A representing a dataset of TESS data cubes for binary classification."""

    def __init__(self, positive_data_directory: str, negative_data_directory: str):
        self.positive_data_directory = positive_data_directory
        self.negative_data_directory = negative_data_directory
        self.time_steps_per_example = 800
        self.batch_size = 100
        training_dataset, testing_dataset = self.generate_datasets()
        self.training_dataset: tf.data.Dataset = training_dataset
        self.testing_dataset: tf.data.Dataset = testing_dataset

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """Generates the training and testing datasets."""
        positive_example_paths = [os.path.join(self.positive_data_directory, file_name) for file_name in
                                  os.listdir(self.positive_data_directory) if file_name.endswith('.npy')]
        positive_example_paths = self.remove_bad_files(positive_example_paths)
        print(f'{len(positive_example_paths)} positive examples.')
        positive_labels = [1] * len(positive_example_paths)
        negative_example_paths = [os.path.join(self.negative_data_directory, file_name) for file_name in
                                  os.listdir(self.negative_data_directory) if file_name.endswith('.npy')]
        negative_example_paths = self.remove_bad_files(negative_example_paths)
        print(f'{len(negative_example_paths)} negative examples.')
        negative_labels = [1] * len(negative_example_paths)
        example_paths = positive_example_paths + negative_example_paths
        labels = positive_labels + negative_labels
        example_paths, labels = self.shuffle_in_unison(example_paths, labels, seed=0)
        file_path_dataset = tf.data.Dataset.from_tensor_slices(example_paths)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        full_dataset = tf.data.Dataset.zip((file_path_dataset, labels_dataset))
        testing_dataset_size = int(len(labels) * 0.2)
        testing_dataset = full_dataset.take(testing_dataset_size)
        training_dataset = full_dataset.skip(testing_dataset_size)
        load_and_preprocess_function = lambda file_path, label: tuple(
            tf.py_function(self.load_and_preprocess_numpy_file, [file_path, label], [tf.float32, tf.int32]))
        training_dataset = training_dataset.map(load_and_preprocess_function)
        training_dataset = training_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        testing_dataset = testing_dataset.map(load_and_preprocess_function)
        testing_dataset = testing_dataset.batch(self.batch_size)
        return training_dataset, testing_dataset

    def load_and_preprocess_numpy_file(self, file_path: tf.Tensor, label: int) -> (np.ndarray, int):
        """Loads numpy files from the tensor alongside labels."""
        cube = np.load(file_path.numpy())
        cube = self.preprocess_cube(cube)
        cube = self.augment_cube(cube)
        return cube.astype(np.float32), label

    def preprocess_cube(self, cube: np.ndarray):
        """Slices and normalizes cubes."""
        time_steps = self.time_steps_per_example
        start_slice = np.random.randint(0, cube.shape[2] - time_steps)
        cube = cube[:, :, start_slice:start_slice + time_steps]
        cube = np.expand_dims(cube, axis=-1)
        cube = np.clip(cube, a_min=1, a_max=None)
        cube = np.log10(cube)
        cube -= np.min(cube)
        array_max = np.max(cube)
        if array_max != 0:
            cube /= array_max
        return cube

    @staticmethod
    def augment_cube(cube: np.ndarray):
        """Augments the data with random rotations and flipping."""
        np.rot90(cube, k=random.randrange(0, 3))
        if random.choice([True, False]):
            np.flip(cube, axis=0)
        if random.choice([True, False]):
            np.flip(cube, axis=1)
        return cube

    @staticmethod
    def remove_bad_files(file_path_list: List[str]):
        """Removes problematic cubes (all values the same, containing infinite or NaN values)."""
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
        return a[indexes], b[indexes]
