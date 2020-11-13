"""
An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections.
"""
import math
from enum import Enum
from functools import partial

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Callable, Tuple

from scipy.interpolate import interp1d

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection
from ramjet.photometric_database.lightcurve_database import LightcurveDatabase
from ramjet.py_mapper import map_py_function_to_dataset


class OutOfBoundsInjectionHandlingMethod(Enum):
    """
    An enum of approaches for handling cases where the injectable signal is shorter than the injectee signal.
    """
    ERROR = 'error'
    REPEAT_SIGNAL = 'repeat_signal'
    RANDOM_INJECTION_LOCATION = 'random_inject_location'


class StandardAndInjectedLightcurveDatabase(LightcurveDatabase):
    """
    An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections
    to be used for training.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections: List[LightcurveCollection] = []
        self.training_injectee_lightcurve_collection: Union[LightcurveCollection, None] = None
        self.training_injectable_lightcurve_collections: List[LightcurveCollection] = []
        self.validation_standard_lightcurve_collections: List[LightcurveCollection] = []
        self.validation_injectee_lightcurve_collection: Union[LightcurveCollection, None] = None
        self.validation_injectable_lightcurve_collections: List[LightcurveCollection] = []
        self.inference_lightcurve_collections: List[LightcurveCollection] = []
        self.shuffle_buffer_size = 10000
        self.number_of_label_types = 1
        self.out_of_bounds_injection_handling: OutOfBoundsInjectionHandlingMethod = \
            OutOfBoundsInjectionHandlingMethod.ERROR

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets for the database.

        :return: The training and validation dataset.
        """
        training_standard_paths_datasets, training_injectee_path_dataset, training_injectable_paths_datasets = \
            self.generate_paths_datasets_group_from_lightcurve_collections_group(
                self.training_standard_lightcurve_collections, self.training_injectee_lightcurve_collection,
                self.training_injectable_lightcurve_collections
            )
        validation_standard_paths_datasets, validation_injectee_path_dataset, validation_injectable_paths_datasets = \
            self.generate_paths_datasets_group_from_lightcurve_collections_group(
                self.validation_standard_lightcurve_collections, self.validation_injectee_lightcurve_collection,
                self.validation_injectable_lightcurve_collections, shuffle=False
            )
        training_lightcurve_and_label_datasets = []
        for paths_dataset, lightcurve_collection in zip(training_standard_paths_datasets,
                                                        self.training_standard_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_standard_lightcurve_and_label_dataset(
                paths_dataset, lightcurve_collection.load_times_fluxes_and_flux_errors_from_path,
                lightcurve_collection.load_label_from_path
            )
            training_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        for paths_dataset, injectable_lightcurve_collection in zip(training_injectable_paths_datasets,
                                                                   self.training_injectable_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_injected_lightcurve_and_label_dataset(
                training_injectee_path_dataset,
                self.training_injectee_lightcurve_collection.load_times_fluxes_and_flux_errors_from_path,
                paths_dataset,
                injectable_lightcurve_collection.load_times_magnifications_and_magnification_errors_from_path,
                injectable_lightcurve_collection.load_label_from_path
            )
            training_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        training_dataset = self.intersperse_datasets(training_lightcurve_and_label_datasets)
        training_dataset = self.window_dataset_for_zipped_example_and_label_dataset(training_dataset, self.batch_size,
                                                                                    self.window_shift)
        validation_lightcurve_and_label_datasets = []
        for paths_dataset, lightcurve_collection in zip(validation_standard_paths_datasets,
                                                        self.validation_standard_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_standard_lightcurve_and_label_dataset(
                paths_dataset, lightcurve_collection.load_times_fluxes_and_flux_errors_from_path,
                lightcurve_collection.load_label_from_path, evaluation_mode=True
            )
            validation_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        for paths_dataset, injectable_lightcurve_collection in zip(validation_injectable_paths_datasets,
                                                                   self.validation_injectable_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_injected_lightcurve_and_label_dataset(
                validation_injectee_path_dataset,
                self.validation_injectee_lightcurve_collection.load_times_fluxes_and_flux_errors_from_path,
                paths_dataset,
                injectable_lightcurve_collection.load_times_magnifications_and_magnification_errors_from_path,
                injectable_lightcurve_collection.load_label_from_path,
                evaluation_mode=True
            )
            validation_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        validation_dataset = self.intersperse_datasets(validation_lightcurve_and_label_datasets)
        validation_dataset = validation_dataset.batch(self.batch_size)
        return training_dataset, validation_dataset

    def generate_paths_datasets_group_from_lightcurve_collections_group(
            self, standard_lightcurve_collections: List[LightcurveCollection],
            injectee_lightcurve_collection: LightcurveCollection,
            injectable_lightcurve_collections: List[LightcurveCollection], shuffle: bool = True
    ) -> (List[tf.data.Dataset], tf.data.Dataset, List[tf.data.Dataset]):
        """
        Create the path dataset for each lightcurve collection in the standard, injectee, and injectable sets.

        :param standard_lightcurve_collections: The standard lightcurve collections.
        :param injectee_lightcurve_collection: The injectee lightcurve collection.
        :param injectable_lightcurve_collections: The injectable lightcurve collections.
        :param shuffle: Whether to shuffle the dataset or not.
        :return: The standard, injectee, and injectable paths datasets.
        """
        injectee_collection_index_in_standard_collection_list: Union[int, None] = None
        for index, standard_lightcurve_collection in enumerate(standard_lightcurve_collections):
            if standard_lightcurve_collection is injectee_lightcurve_collection:
                injectee_collection_index_in_standard_collection_list = index
        if injectee_collection_index_in_standard_collection_list is not None:
            standard_lightcurve_collections.pop(injectee_collection_index_in_standard_collection_list)
        standard_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            standard_lightcurve_collections, shuffle=shuffle)
        injectee_path_dataset = None
        if injectee_lightcurve_collection is not None:
            injectee_path_dataset = self.generate_paths_dataset_from_lightcurve_collection(
                injectee_lightcurve_collection, shuffle=shuffle)
            number_of_elements_repeated_in_a_row = len(injectable_lightcurve_collections)
            if injectee_collection_index_in_standard_collection_list is not None:
                number_of_elements_repeated_in_a_row += 1
            injectee_path_dataset = injectee_path_dataset.flat_map(
                partial(repeat_each_element, number_of_repeats=number_of_elements_repeated_in_a_row))
            if injectee_collection_index_in_standard_collection_list is not None:
                standard_paths_datasets.insert(injectee_collection_index_in_standard_collection_list,
                                               injectee_path_dataset)
        injectable_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            injectable_lightcurve_collections, shuffle=shuffle)
        return standard_paths_datasets, injectee_path_dataset, injectable_paths_datasets

    def generate_paths_dataset_from_lightcurve_collection(self, lightcurve_collection: LightcurveCollection,
                                                          repeat: bool = True, shuffle: bool = True
                                                          ) -> tf.data.Dataset:
        """
        Generates a paths dataset for a lightcurve collection.

        :param lightcurve_collection: The lightcurve collection to generate a paths dataset for.
        :param repeat: Whether to repeat the dataset or not.
        :param shuffle: Whether to shuffle the dataset or not.
        :return: The paths dataset.
        """
        dataset = self.paths_dataset_from_list_or_generator_factory(lightcurve_collection.get_paths)
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        return dataset

    def generate_paths_datasets_from_lightcurve_collection_list(self,
                                                                lightcurve_collections: List[LightcurveCollection],
                                                                shuffle: bool = True) -> List[tf.data.Dataset]:
        """
        Generates a paths dataset for each lightcurve collection in a list.

        :param lightcurve_collections: The list of lightcurve collections.
        :param shuffle: Whether to shuffle the datasets or not.
        :return: The list of paths datasets.
        """
        return [self.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection, shuffle=shuffle)
                for lightcurve_collection in lightcurve_collections]

    def generate_standard_lightcurve_and_label_dataset(
            self, paths_dataset: tf.data.Dataset,
            load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]], evaluation_mode: bool = False):
        """
        Generates a lightcurve and label dataset from a paths dataset using a passed function defining
        how to load the values from the lightcurve file and the label value to use.

        :param paths_dataset: The dataset of paths to use.
        :param load_times_fluxes_and_flux_errors_from_path_function: The function defining how to load the times and
                                                                     fluxes of a lightcurve from a path.
        :param load_label_from_path_function: The function to load the label to use for the lightcurves in this dataset.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :return: The resulting lightcurve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_standard_lightcurve,
                                          load_times_fluxes_and_flux_errors_from_path_function,
                                          load_label_from_path_function,
                                          evaluation_mode=evaluation_mode)
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (self.number_of_label_types,)]
        example_and_label_dataset = map_py_function_to_dataset(paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_standard_lightcurve(
            self,
            load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]],
            lightcurve_path_tensor: tf.Tensor, evaluation_mode: bool = False) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual standard lightcurve from a lightcurve path tensor, using a passed function defining
        how to load the values from the lightcurve file and the label value to use. Designed to be used with `partial`
        to prepare a function which will just require the lightcurve path tensor, and can then be mapped to a dataset.

        :param load_times_fluxes_and_flux_errors_from_path_function: The function to load the lightcurve times and
                                                                     fluxes from a file.
        :param load_label_from_path_function: The function to load the label to assign to the lightcurve.
        :param lightcurve_path_tensor: The tensor containing the path to the lightcurve file.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :return: The example and label arrays shaped for use as single example for the network.
        """
        lightcurve_path = Path(lightcurve_path_tensor.numpy().decode('utf-8'))
        times, fluxes, flux_errors = load_times_fluxes_and_flux_errors_from_path_function(lightcurve_path)
        if flux_errors is not None:
            raise NotImplementedError
        light_curve = self.build_light_curve_array(fluxes=fluxes, times=times)
        example = self.preprocess_light_curve(light_curve, evaluation_mode=evaluation_mode)
        label = load_label_from_path_function(lightcurve_path)
        label = self.expand_label_to_training_dimensions(label)
        return example, label

    @staticmethod
    def expand_label_to_training_dimensions(label: Union[int, List[int], Tuple[int], np.ndarray]) -> np.ndarray:
        """
        Expand the label to the appropriate dimensions for training.

        :param label: The label to convert.
        :return: The label with the correct dimensions.
        """
        if type(label) is not np.ndarray:
            if type(label) in [list, tuple]:
                label = np.array(label)
            else:
                label = np.array([label])
        return label

    def generate_infer_path_and_lightcurve_dataset(
            self, paths_dataset: tf.data.Dataset,
            load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]]):
        """
        Generates a path and lightcurve dataset from a paths dataset using a passed function defining
        how to load the values from the lightcurve file.

        :param paths_dataset: The dataset of paths to use.
        :param load_times_fluxes_and_flux_errors_from_path_function: The function defining how to load the times and
                                                                     fluxes of a lightcurve from a path.
        :return: The resulting lightcurve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_infer_lightcurve,
                                          load_times_fluxes_and_flux_errors_from_path_function)
        output_types = (tf.string, tf.float32)
        output_shapes = [(), (self.time_steps_per_example, 1)]
        example_and_label_dataset = map_py_function_to_dataset(paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_infer_lightcurve(
            self, load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            lightcurve_path_tensor: tf.Tensor) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual standard lightcurve from a lightcurve path tensor, using a passed function defining
        how to load the values from the lightcurve file and returns the path and lightcurve. Designed to be used with
        `partial` to prepare a function which will just require the lightcurve path tensor, and can then be mapped to a
        dataset.

        :param load_times_fluxes_and_flux_errors_from_path_function: The function to load the lightcurve times and
                                                                     fluxes from a file.
        :param lightcurve_path_tensor: The tensor containing the path to the lightcurve file.
        :return: The path and example array shaped for use as single example for the network.
        """
        lightcurve_path_string = lightcurve_path_tensor.numpy().decode('utf-8')
        lightcurve_path = Path(lightcurve_path_string)
        times, fluxes, flux_errors = load_times_fluxes_and_flux_errors_from_path_function(lightcurve_path)
        if flux_errors is not None:
            raise NotImplementedError
        light_curve = self.build_light_curve_array(fluxes=fluxes, times=times)
        example = self.preprocess_light_curve(light_curve, evaluation_mode=True)
        return lightcurve_path_string, example

    def generate_injected_lightcurve_and_label_dataset(
            self, injectee_paths_dataset: tf.data.Dataset,
            injectee_load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            injectable_paths_dataset: tf.data.Dataset,
            injectable_load_times_magnifications_and_magnification_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]], evaluation_mode: bool = False):
        """
        Generates a lightcurve and label dataset from an injectee and injectable paths dataset, using passed functions
        defining how to load the values from the lightcurve files for each and the label value to use.

        :param injectee_paths_dataset: The dataset of paths to use for the injectee lightcurves.
        :param injectee_load_times_fluxes_and_flux_errors_from_path_function: The function defining how to load the
            times and fluxes of an injectee lightcurve from a path.
        :param injectable_paths_dataset: The dataset of paths to use for the injectable lightcurves.
        :param injectable_load_times_magnifications_and_magnification_errors_from_path_function: The function defining
            how to load the times and magnifications of an injectable signal from a path.
        :param load_label_from_path_function: The function to load the label to use for the lightcurves in this dataset.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :return: The resulting lightcurve example and label dataset.
        """
        preprocess_map_function = partial(
            self.preprocess_injected_lightcurve,
            injectee_load_times_fluxes_and_flux_errors_from_path_function,
            injectable_load_times_magnifications_and_magnification_errors_from_path_function,
            load_label_from_path_function,
            evaluation_mode=evaluation_mode)
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (self.number_of_label_types,)]
        zipped_paths_dataset = tf.data.Dataset.zip((injectee_paths_dataset, injectable_paths_dataset))
        example_and_label_dataset = map_py_function_to_dataset(zipped_paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_injected_lightcurve(
            self,
            injectee_load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            injectable_load_times_magnifications_and_magnification_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]],
            injectee_lightcurve_path_tensor: tf.Tensor, injectable_lightcurve_path_tensor: tf.Tensor,
            evaluation_mode: bool = False
    ) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual injected lightcurve from an injectee and an injectable lightcurve path tensor,
        using a passed function defining how to load the values from each lightcurve file and the label value to use.
        Designed to be used with `partial` to prepare a function which will just require the lightcurve path tensor, and
        can then be mapped to a dataset.

        :param injectee_load_times_fluxes_and_flux_errors_from_path_function: The function to load the injectee
            lightcurve times and fluxes from a file.
        :param injectable_load_times_magnifications_and_magnification_errors_from_path_function: The function to load
            the injectee lightcurve times and signal from a file.
        :param load_label_from_path_function: The function to load the label to assign to the lightcurve.
        :param injectee_lightcurve_path_tensor: The tensor containing the path to the injectee lightcurve file.
        :param injectable_lightcurve_path_tensor: The tensor containing the path to the injectable lightcurve file.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :return: The injected example and label arrays shaped for use as single example for the network.
        """
        injectee_lightcurve_path = Path(injectee_lightcurve_path_tensor.numpy().decode('utf-8'))
        injectee_arrays = injectee_load_times_fluxes_and_flux_errors_from_path_function(injectee_lightcurve_path)
        injectee_times, injectee_fluxes, injectee_flux_errors = injectee_arrays
        injectable_lightcurve_path = Path(injectable_lightcurve_path_tensor.numpy().decode('utf-8'))
        injectable_arrays = injectable_load_times_magnifications_and_magnification_errors_from_path_function(
            injectable_lightcurve_path)
        injectable_times, injectable_magnifications, injectable_magnification_errors = injectable_arrays
        if injectee_flux_errors is not None or injectable_magnification_errors is not None:
            raise NotImplementedError
        fluxes = self.inject_signal_into_lightcurve(injectee_fluxes, injectee_times, injectable_magnifications,
                                                    injectable_times)
        light_curve = self.build_light_curve_array(fluxes=fluxes, times=injectee_times)
        example = self.preprocess_light_curve(light_curve, evaluation_mode=evaluation_mode)
        label = load_label_from_path_function(injectable_lightcurve_path)
        label = self.expand_label_to_training_dimensions(label)
        return example, label

    def inject_signal_into_lightcurve(self, lightcurve_fluxes: np.ndarray, lightcurve_times: np.ndarray,
                                      signal_magnifications: np.ndarray, signal_times: np.ndarray):
        """
        Injects a synthetic magnification signal into real lightcurve fluxes.

        :param lightcurve_fluxes: The fluxes of the lightcurve to be injected into.
        :param lightcurve_times: The times of the flux observations of the lightcurve.
        :param signal_magnifications: The synthetic magnifications to inject.
        :param signal_times: The times of the synthetic magnifications.
        :return: The fluxes with the injected signal.
        """
        minimum_lightcurve_time = np.min(lightcurve_times)
        relative_lightcurve_times = lightcurve_times - minimum_lightcurve_time
        relative_signal_times = signal_times - np.min(signal_times)
        signal_time_length = np.max(relative_signal_times)
        lightcurve_time_length = np.max(relative_lightcurve_times)
        time_length_difference = lightcurve_time_length - signal_time_length
        signal_start_offset = (np.random.random() * time_length_difference) + minimum_lightcurve_time
        offset_signal_times = relative_signal_times + signal_start_offset
        median_flux = np.median(lightcurve_fluxes)
        signal_fluxes = (signal_magnifications * median_flux) - median_flux
        if self.out_of_bounds_injection_handling is OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION:
            signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=False, fill_value=0)
        elif (self.out_of_bounds_injection_handling is OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL and
              time_length_difference > 0):
            before_signal_gap = signal_start_offset - minimum_lightcurve_time
            after_signal_gap = time_length_difference - before_signal_gap
            minimum_signal_time_step = np.min(np.diff(offset_signal_times))
            before_repeats_needed = math.ceil(before_signal_gap / (signal_time_length + minimum_signal_time_step))
            after_repeats_needed = math.ceil(after_signal_gap / (signal_time_length + minimum_signal_time_step))
            repeated_signal_fluxes = np.tile(signal_fluxes, before_repeats_needed + 1 + after_repeats_needed)
            repeated_signal_times = None
            for repeat_index in range(-before_repeats_needed, after_repeats_needed + 1):
                repeat_signal_start_offset = (signal_time_length + minimum_signal_time_step) * repeat_index
                if repeated_signal_times is None:
                    repeated_signal_times = offset_signal_times + repeat_signal_start_offset
                else:
                    repeat_index_signal_times = offset_signal_times + repeat_signal_start_offset
                    repeated_signal_times = np.concatenate([repeated_signal_times, repeat_index_signal_times])
            signal_flux_interpolator = interp1d(repeated_signal_times, repeated_signal_fluxes, bounds_error=True)
        else:
            signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=True)
        interpolated_signal_fluxes = signal_flux_interpolator(lightcurve_times)
        fluxes_with_injected_signal = lightcurve_fluxes + interpolated_signal_fluxes
        return fluxes_with_injected_signal

    @staticmethod
    def intersperse_datasets(dataset_list: List[tf.data.Dataset]) -> tf.data.Dataset:
        """
        Intersperses a list of datasets into one joint dataset. (e.g., [0, 2, 4] and [1, 3, 5] to [0, 1, 2, 3, 4, 5]).

        :param dataset_list: The datasets to intersperse.
        :return: The interspersed dataset.
        """
        dataset_tuple = tuple(dataset_list)
        zipped_dataset = tf.data.Dataset.zip(dataset_tuple)

        def flat_map_interspersing_function(*elements):
            """Intersperses an individual element from each dataset. To be used by flat_map."""
            concatenated_element = tf.data.Dataset.from_tensors(elements[0])
            for element in elements[1:]:
                concatenated_element = concatenated_element.concatenate(tf.data.Dataset.from_tensors(element))
            return concatenated_element

        flat_mapped_dataset = zipped_dataset.flat_map(flat_map_interspersing_function)
        return flat_mapped_dataset

    def generate_inference_dataset(self):
        """
        Generates the dataset to infer over.

        :return: The inference dataset.
        """
        batch_dataset = None
        for lightcurve_collection in self.inference_lightcurve_collections:
            example_paths_dataset = self.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection,
                                                                                           repeat=False, shuffle=False)
            examples_dataset = self.generate_infer_path_and_lightcurve_dataset(
                example_paths_dataset, lightcurve_collection.load_times_fluxes_and_flux_errors_from_path)
            collection_batch_dataset = examples_dataset.batch(self.batch_size)
            if batch_dataset is None:
                batch_dataset = collection_batch_dataset
            else:
                batch_dataset = batch_dataset.concatenate(collection_batch_dataset)
        batch_dataset = batch_dataset.prefetch(5)
        return batch_dataset


def repeat_each_element(element: tf.Tensor, number_of_repeats: int) -> tf.data.Dataset:
    """
    A dataset mappable function which repeats the elements a given number of times.

    :param element: The element to map to to repeat.
    :param number_of_repeats: The number of times to repeat the element.
    :return: The dataset with repeated elements.
    """
    return tf.data.Dataset.from_tensors(element).repeat(number_of_repeats)
