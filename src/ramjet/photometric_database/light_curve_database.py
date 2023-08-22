"""Code for a base generalized database for photometric data to be subclassed."""
import math
import shutil
from abc import ABC
from pathlib import Path
from typing import List, Union, Callable, Iterable, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf


def preprocess_times(light_curve_array: np.ndarray) -> None:
    """
    Preprocesses the times of the light curve.

    :param light_curve_array: The light curve array to preprocess.
    :return: The light curve array with the times preprocessed.
    """
    times = light_curve_array[:, 0]
    light_curve_array[:, 0] = calculate_time_differences(times)


def make_times_and_fluxes_array_uniform_length(arrays: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], length: int, randomize: bool = True) -> (np.ndarray, np.ndarray):
    times, fluxes = arrays
    light_curve_array = np.stack([times, fluxes], axis=-1)
    uniform_length_light_curve_array = make_uniform_length(light_curve_array, length=length, randomize=randomize)
    return uniform_length_light_curve_array[:, 0], uniform_length_light_curve_array[:, 1]


def make_times_and_label_array_uniform_length(arrays: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], length: int, randomize: bool = True) -> (np.ndarray, np.ndarray):
    times, label = arrays
    uniform_length_times = make_uniform_length(times, length=length, randomize=randomize)
    return uniform_length_times, label


def make_uniform_length(example: np.ndarray, length: int, randomize: bool = True) -> np.ndarray:
    """Makes the example a specific length, by clipping those too large and repeating those too small."""
    assert len(example.shape) in [1, 2]  # Only tested for 1D and 2D cases.
    if randomize:
        example = randomly_roll_elements(example)
    if example.shape[0] == length:
        pass
    elif example.shape[0] > length:
        example = example[:length]
    else:
        elements_to_repeat = length - example.shape[0]
        if len(example.shape) == 1:
            example = np.pad(example, (0, elements_to_repeat), mode='wrap')
        else:
            example = np.pad(example, ((0, elements_to_repeat), (0, 0)), mode='wrap')
    return example


def get_ratio_enforced_dataset(positive_training_dataset: tf.data.Dataset,
                               negative_training_dataset: tf.data.Dataset,
                               positive_to_negative_data_ratio: float) -> tf.data.Dataset:
    """Generates a dataset with an enforced data ratio."""
    if positive_to_negative_data_ratio is not None:
        positive_count = len(list(positive_training_dataset))
        negative_count = len(list(negative_training_dataset))
        existing_ratio = positive_count / negative_count
        if existing_ratio < positive_to_negative_data_ratio:
            desired_number_of_positive_examples = int(positive_to_negative_data_ratio * negative_count)
            positive_training_dataset = repeat_dataset_to_size(positive_training_dataset,
                                                               desired_number_of_positive_examples)
        else:
            desired_number_of_negative_examples = int((1 / positive_to_negative_data_ratio) * positive_count)
            negative_training_dataset = repeat_dataset_to_size(negative_training_dataset,
                                                               desired_number_of_negative_examples)
    return positive_training_dataset.concatenate(negative_training_dataset)


class LightCurveDatabase(ABC):
    """A base generalized database for photometric data to be subclassed."""

    def __init__(self, data_directory='data'):
        self.time_steps_per_example = 16000
        self.data_directory: Path = Path(data_directory)
        self.validation_ratio: float = 0.2
        self.batch_size: int = 100
        self.time_steps_per_example: int
        self.number_of_parallel_processes_per_map: int = 16
        self.include_time_as_channel: bool = False
        self.include_flux_errors_as_channel: bool = False

    @property
    def window_shift(self) -> int:
        """
        How much the window shifts for a windowed batch set.

        :return: The window shift size.
        """
        return self.batch_size // 10

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

        example_paths_dataset = paths_dataset_from_list_or_generator_factory(example_paths)
        training_example_paths_dataset = example_paths_dataset.enumerate().filter(element_should_be_in_training
                                                                                  ).map(drop_index)
        validation_example_paths_dataset = example_paths_dataset.enumerate().filter(element_should_be_in_validation
                                                                                    ).map(drop_index)
        return training_example_paths_dataset, validation_example_paths_dataset

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

    def normalize_fluxes(self, light_curve: np.ndarray) -> None:
        """
        Normalizes the flux channel of the light curve in-place.

        :param light_curve: The light curve whose flux channel should be normalized.
        :return: The light curve with the flux channel normalized.
        """
        if self.include_time_as_channel:
            if self.include_flux_errors_as_channel:
                assert light_curve.shape[1] == 3
                light_curve[:, 1], light_curve[:, 2] = normalize_on_percentiles_with_errors(
                    light_curve[:, 1], light_curve[:, 2])
            else:
                assert light_curve.shape[1] == 2
                light_curve[:, 1] = normalize_on_percentiles(light_curve[:, 1])
        else:
            assert light_curve.shape[1] == 1
            light_curve[:, 0] = normalize_on_percentiles(light_curve[:, 0])

    def build_light_curve_array(self, fluxes: np.ndarray, times: Union[np.ndarray, None] = None,
                                flux_errors: Union[np.ndarray, None] = None):
        """
        Builds the light curve array based on the components required for the specific database setup.

        :param fluxes: The fluxes of the light curve.
        :param times: The optional times of the light curve.
        :param flux_errors: The optional flux errors of the light curve.
        :return: The constructed light curve array.
        """
        if self.include_flux_errors_as_channel:
            if not self.include_time_as_channel:
                raise NotImplementedError
            light_curve = np.stack([times, fluxes, flux_errors], axis=-1)
        elif self.include_time_as_channel:
            light_curve = np.stack([times, fluxes], axis=-1)
        else:
            light_curve = np.expand_dims(fluxes, axis=-1)
        return light_curve

    def preprocess_light_curve(self, light_curve: np.ndarray, evaluation_mode: bool = False) -> np.ndarray:
        """
        Preprocessing for the light curve.

        :param light_curve: The light curve array to preprocess.
        :param evaluation_mode: If the preprocessing should be consistent for evaluation.
        :return: The preprocessed flux array.
        """
        if not evaluation_mode:
            light_curve = remove_random_elements(light_curve)
        light_curve = make_uniform_length(light_curve, self.time_steps_per_example,
                                          randomize=not evaluation_mode)
        self.normalize_fluxes(light_curve)
        if self.include_time_as_channel:
            preprocess_times(light_curve)
        return light_curve


def normalize_log_0_to_1(light_curve: np.ndarray) -> np.ndarray:
    """Normalizes from 0 to 1 on the logarithm of the light curve."""
    light_curve -= np.min(light_curve)
    light_curve = np.log1p(light_curve)
    array_max = np.max(light_curve)
    if array_max != 0:
        light_curve /= array_max
    return light_curve


def normalize_on_percentiles(array: np.ndarray) -> np.ndarray:
    """
    Normalizes an array using percentiles. The 10th percentile is normalized to -1, the 90th to 1.

    :param array: The array to be normalized.
    :return: The normalized array.
    """
    percentile_10 = np.percentile(array, 10)
    percentile_90 = np.percentile(array, 90)
    percentile_difference = percentile_90 - percentile_10
    if percentile_difference == 0:
        normalized_array = np.zeros_like(array)
    else:
        normalized_array = ((array - percentile_10) / (percentile_difference / 2)) - 1
    return normalized_array


def normalize_on_percentiles_with_errors(array: np.ndarray, array_errors: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Normalizes an array using percentiles. The 10th percentile is normalized to -1, the 90th to 1.
    Scales the errors by the corresponding scaling factor.
    """
    percentile_10 = np.percentile(array, 10)
    percentile_90 = np.percentile(array, 90)
    percentile_difference = percentile_90 - percentile_10
    if percentile_difference == 0:
        normalized_array = np.zeros_like(array)
        normalized_array_errors = np.zeros_like(array_errors)
    else:
        normalized_array = ((array - percentile_10) / (percentile_difference / 2)) - 1
        normalized_array_errors = array_errors / (percentile_difference / 2)
    return normalized_array, normalized_array_errors


def shuffle_in_unison(a, b, seed=None):
    """Shuffle two arrays in unison."""
    if seed is not None:
        np.random.seed(seed)
    indexes = np.random.permutation(len(a))
    return np.array(a)[indexes], np.array(b)[indexes]


def remove_random_elements(light_curve: np.ndarray, ratio: float = 0.01) -> np.ndarray:
    """Removes random values from the light_curve."""
    light_curve_length = light_curve.shape[0]
    max_values_to_remove = int(light_curve_length * ratio)
    if max_values_to_remove != 0:
        values_to_remove = np.random.randint(max_values_to_remove)
    else:
        values_to_remove = 0
    random_indexes = np.random.choice(range(light_curve_length), values_to_remove, replace=False)
    return np.delete(light_curve, random_indexes, axis=0)


def repeat_dataset_to_size(dataset: tf.data.Dataset, size: int) -> tf.data.Dataset:
    """Repeats a dataset to make it a desired length."""
    current_size = len(list(dataset))
    times_to_repeat = math.ceil(size / current_size)
    return dataset.repeat(times_to_repeat).take(size)


def randomly_roll_elements(example: np.ndarray) -> np.ndarray:
    """Randomly rolls the elements."""
    example = np.roll(example, np.random.randint(example.shape[0]), axis=0)
    return example


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


def calculate_time_differences(times: np.ndarray) -> np.ndarray:
    """
    Calculates the differences between an array of time, doubling up the first element to make the length the same.

    :param times: The times to difference.
    :return: The time differences.
    """
    difference_times = np.diff(times)
    difference_times = np.insert(difference_times, 0, difference_times[0], axis=0)
    return difference_times
