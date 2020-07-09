"""Code for a base generalized database for photometric data to be subclassed."""
import math
import shutil
from abc import ABC
import os
import random
from pathlib import Path
from typing import List, Union, Tuple, Callable, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf


class LightcurveDatabase(ABC):
    """A base generalized database for photometric data to be subclassed."""

    def __init__(self, data_directory='data'):
        self.data_directory: Path = Path(data_directory)
        self.validation_ratio = 0.2
        self.batch_size = 100
        self.trial_directory = None
        self.time_steps_per_example: int
        self.number_of_parallel_processes_per_map = 16

    @property
    def window_shift(self) -> int:
        """
        How much the window shifts for a windowed batch set.

        :return: The window shift size.
        """
        return self.batch_size // 10

    def log_dataset_file_names(self, dataset: tf.data.Dataset, dataset_name: str):
        """Saves the names of the files used in a dataset to a CSV file in the trail directory."""
        os.makedirs(self.trial_directory, exist_ok=True)
        training_example_paths = [example.numpy().decode('utf-8') for example in list(dataset)]
        series = pd.Series(training_example_paths)
        series.to_csv(os.path.join(self.trial_directory, f'{dataset_name}.csv'), header=False, index=False)

    @staticmethod
    def normalize_log_0_to_1(lightcurve: np.ndarray) -> np.ndarray:
        """Normalizes from 0 to 1 on the logarithm of the lightcurve."""
        lightcurve -= np.min(lightcurve)
        lightcurve = np.log1p(lightcurve)
        array_max = np.max(lightcurve)
        if array_max != 0:
            lightcurve /= array_max
        return lightcurve

    def normalize(self, lightcurve: np.ndarray) -> np.ndarray:
        """
        Normalizes the lightcurve.

        :param lightcurve: The lightcurve to normalize.
        :return: The normalized lightcurve.
        """
        return self.normalize_on_percentiles(lightcurve)

    @staticmethod
    def normalize_on_percentiles(lightcurve: np.ndarray) -> np.ndarray:
        """
        Normalizes light curve using percentiles. The 10th percentile is normalized to -1, the 90th to 1.
        """
        percentile_10 = np.percentile(lightcurve, 10)
        percentile_90 = np.percentile(lightcurve, 90)
        percentile_difference = percentile_90 - percentile_10
        if percentile_difference == 0:
            normalized_lightcurve = np.zeros_like(lightcurve)
        else:
            normalized_lightcurve = ((lightcurve - percentile_10) / (percentile_difference / 2)) - 1
        return normalized_lightcurve

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
    def make_uniform_length(example: np.ndarray, length: int, randomize: bool = True, seed: int = None) -> np.ndarray:
        """Makes the example a specific length, by clipping those too large and repeating those too small."""
        if seed is not None:
            np.random.seed(seed)
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
            if randomize:
                pre_padding = np.random.randint(0, elements_to_repeat + 1)
            else:
                pre_padding = 0
            post_padding = elements_to_repeat - pre_padding
            if len(example.shape) == 2:
                example = np.pad(example, ((pre_padding, post_padding), (0, 0)), mode='wrap')
            else:
                example = np.pad(example, (pre_padding, post_padding), mode='wrap')
        return example

    def get_training_and_validation_datasets_for_file_paths(
                self, example_paths: Union[Iterable[Path], Callable[[], Iterable[Path]]]
            ) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Creates training and validation datasets from a list of all file paths. The database validation ratio is used
        to determine the size of the split.

        :param example_paths: The total list of file paths.
        :return: The training and validation datasets.
        """
        def element_should_be_in_validation(index, _):
            """Checks if the element should be in the validation set based on the index."""
            return index % int(1 / self.validation_ratio) == 0

        def element_should_be_in_training(index, element):
            """Checks if the element should be in the training set based on the index."""
            return not element_should_be_in_validation(index, element)

        def drop_index(_, element):
            """Drops the index from the index element pair dataset."""
            return element

        example_paths_dataset = self.paths_dataset_from_list_or_generator_factory(example_paths)
        training_example_paths_dataset = example_paths_dataset.enumerate().filter(element_should_be_in_training
                                                                                  ).map(drop_index)
        validation_example_paths_dataset = example_paths_dataset.enumerate().filter(element_should_be_in_validation
                                                                                    ).map(drop_index)
        return training_example_paths_dataset, validation_example_paths_dataset

    @staticmethod
    def extract_shuffled_chunk_and_remainder(array_to_extract_from: Union[List, np.ndarray], chunk_ratio: float,
                                             chunk_to_extract_index: int = 0) -> (np.ndarray, np.ndarray):
        """
        Shuffles an array, extracts a chunk of the data, and returns the chunk and remainder of the array.

        :param array_to_extract_from: The array to process.
        :param chunk_ratio: The number of equal size chunks to split the array into before extracting one.
        :param chunk_to_extract_index: The index of the chunk to extract out of all chunks.
        :return: The chunk which is extracted, and the remainder of the array excluding the chunk.
        """
        np.random.seed(0)
        np.random.shuffle(array_to_extract_from)
        number_of_chunks = int(1 / chunk_ratio)
        chunks = np.array_split(array_to_extract_from, number_of_chunks)
        extracted_chunk = chunks[chunk_to_extract_index]
        remaining_chunks = np.delete(chunks, chunk_to_extract_index, axis=0)
        remainder = np.concatenate(remaining_chunks)
        return extracted_chunk, remainder

    def flat_window_zipped_example_and_label_dataset(self, dataset, batch_size, window_shift):
        """
        Takes a zipped example and label dataset and repeats examples in a windowed fashion of a given batch size.
        It is expected that the resulting dataset will subsequently be batched in some fashion by the given batch size.

        :param dataset: The zipped example and label dataset.
        :param batch_size: The size of the batches to produce.
        :param window_shift: The shift of the moving window between batches.
        :return: The flattened window dataset.
        """
        examples_dataset = dataset.map(lambda element, _: element)
        labels_dataset = dataset.map(lambda _, element: element)
        examples_window_dataset = examples_dataset.window(batch_size, shift=window_shift)
        examples_unbatched_window_dataset = examples_window_dataset.flat_map(lambda element: element)
        labels_window_dataset = labels_dataset.window(batch_size, shift=window_shift)
        labels_unbatched_window_dataset = labels_window_dataset.flat_map(lambda element: element)
        unbatched_window_dataset = tf.data.Dataset.zip((examples_unbatched_window_dataset,
                                                        labels_unbatched_window_dataset))
        return unbatched_window_dataset

    def padded_window_dataset_for_zipped_example_and_label_dataset(self, dataset: tf.data.Dataset, batch_size: int,
                                                                   window_shift: int,
                                                                   padded_shapes: Tuple[List, List]) -> tf.data.Dataset:
        """
        Takes a zipped example and label dataset, and converts it to padded batches, where each batch uses overlapping
        examples based on a sliding window.

        :param dataset: The zipped example and label dataset.
        :param batch_size: The size of the batches to produce.
        :param window_shift: The shift of the moving window between batches.
        :param padded_shapes: The output padded shape.
        :return: The padded window dataset.
        """
        unbatched_window_dataset = self.flat_window_zipped_example_and_label_dataset(dataset, batch_size,
                                                                                     window_shift)
        return unbatched_window_dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

    def window_dataset_for_zipped_example_and_label_dataset(self, dataset: tf.data.Dataset, batch_size: int,
                                                            window_shift: int) -> tf.data.Dataset:
        """
        Takes a zipped example and label dataset, and converts it to batches, where each batch uses overlapping
        examples based on a sliding window.

        :param dataset: The zipped example and label dataset.
        :param batch_size: The size of the batches to produce.
        :param window_shift: The shift of the moving window between batches.
        :return: The window dataset.
        """
        unbatched_window_dataset = self.flat_window_zipped_example_and_label_dataset(dataset, batch_size,
                                                                                     window_shift)
        return unbatched_window_dataset.batch(batch_size)

    def clear_data_directory(self):
        """
        Empties the data directory.
        """
        if self.data_directory.exists():
            shutil.rmtree(self.data_directory)
        self.create_data_directories()

    def create_data_directories(self):
        """
        Creates the data directories to be used by the database.
        """
        self.data_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def paths_dataset_from_list_or_generator_factory(
            list_or_generator_factory: Union[Iterable[Path], Callable[[], Iterable[Path]]]
            ) -> tf.data.Dataset:
        """
        Produces a dataset from either the examples path list or example paths factory to strings.

        :param list_or_generator_factory: The list or generator factory.
        :return: The new path generator.
        """
        def paths_to_strings_generator():
            """A generator from either the examples path list or example paths factory to strings."""
            if isinstance(list_or_generator_factory, Callable):  # If factory, produce a new generator/list.
                resolved_example_paths = list_or_generator_factory()
            else:  # Otherwise, the paths are already a resolved list, and can be directly used.
                resolved_example_paths = list_or_generator_factory
            for path in resolved_example_paths:
                yield str(path)
        return tf.data.Dataset.from_generator(paths_to_strings_generator, output_types=tf.string)
