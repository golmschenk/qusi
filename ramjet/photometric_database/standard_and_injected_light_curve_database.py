"""
An abstract class allowing for any number and combination of standard and injectable/injectee light curve collections.
"""
import math
from enum import Enum
from functools import partial
from queue import Queue

import numpy as np
import scipy.stats
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Callable, Tuple, Optional
from scipy.interpolate import interp1d

from ramjet.logging.wandb_logger import WandbLogger, WandbLoggableLightCurve, \
    WandbLoggableInjection
from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.light_curve_collection import LightCurveCollection
from ramjet.photometric_database.light_curve_database import LightCurveDatabase
from ramjet.py_mapper import map_py_function_to_dataset


class OutOfBoundsInjectionHandlingMethod(Enum):
    """
    An enum of approaches for handling cases where the injectable signal is shorter than the injectee signal.
    """
    ERROR = 'error'
    REPEAT_SIGNAL = 'repeat_signal'
    RANDOM_INJECTION_LOCATION = 'random_inject_location'


class BaselineFluxEstimationMethod(Enum):
    """
    An enum of to designate the type of baseline flux estimation method to use during training.
    """
    MEDIAN = 'median'
    MEDIAN_ABSOLUTE_DEVIATION = 'median_absolute_deviation'


class StandardAndInjectedLightCurveDatabase(LightCurveDatabase):
    """
    An abstract class allowing for any number and combination of standard and injectable/injectee light curve collections
    to be used for training.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections: List[LightCurveCollection] = []
        self.training_injectee_light_curve_collection: Union[LightCurveCollection, None] = None
        self.training_injectable_light_curve_collections: List[LightCurveCollection] = []
        self.validation_standard_light_curve_collections: List[LightCurveCollection] = []
        self.validation_injectee_light_curve_collection: Union[LightCurveCollection, None] = None
        self.validation_injectable_light_curve_collections: List[LightCurveCollection] = []
        self.inference_light_curve_collections: List[LightCurveCollection] = []
        self.shuffle_buffer_size = 10000
        self.number_of_label_types = 1
        self.out_of_bounds_injection_handling: OutOfBoundsInjectionHandlingMethod = \
            OutOfBoundsInjectionHandlingMethod.ERROR
        self.baseline_flux_estimation_method = BaselineFluxEstimationMethod.MEDIAN
        self.logger: Optional[WandbLogger] = None

    @property
    def number_of_input_channels(self) -> int:
        """
        Determines the number of input channels that should exist for this database.

        :return: The number of channels.
        """
        channels = 1
        if self.include_time_as_channel:
            channels += 1
        if self.include_flux_errors_as_channel:
            channels += 1
        return channels

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets for the database.

        :return: The training and validation dataset.
        """
        training_standard_paths_datasets, training_injectee_path_dataset, training_injectable_paths_datasets = \
            self.generate_paths_datasets_group_from_light_curve_collections_group(
                self.training_standard_light_curve_collections, self.training_injectee_light_curve_collection,
                self.training_injectable_light_curve_collections)
        validation_standard_paths_datasets, validation_injectee_path_dataset, validation_injectable_paths_datasets = \
            self.generate_paths_datasets_group_from_light_curve_collections_group(
                self.validation_standard_light_curve_collections, self.validation_injectee_light_curve_collection,
                self.validation_injectable_light_curve_collections, shuffle=False)
        training_light_curve_and_label_datasets = []
        for index, (paths_dataset, light_curve_collection) in enumerate(
                zip(training_standard_paths_datasets, self.training_standard_light_curve_collections)):
            light_curve_and_label_dataset = self.generate_standard_light_curve_and_label_dataset(paths_dataset,
                                                                                                 light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                                                                                                 light_curve_collection.load_label_from_path,
                                                                                                 name=f"{type(light_curve_collection).__name__}_standard_train_{index}")
            training_light_curve_and_label_datasets.append(light_curve_and_label_dataset)
        for index, (paths_dataset, injectable_light_curve_collection) in enumerate(
                zip(training_injectable_paths_datasets, self.training_injectable_light_curve_collections)):
            light_curve_and_label_dataset = self.generate_injected_light_curve_and_label_dataset(
                training_injectee_path_dataset,
                self.training_injectee_light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                paths_dataset,
                injectable_light_curve_collection.load_times_magnifications_and_magnification_errors_from_path,
                injectable_light_curve_collection.load_label_from_path,
                name=f"{type(injectable_light_curve_collection).__name__}_injected_train_{index}")
            training_light_curve_and_label_datasets.append(light_curve_and_label_dataset)
        training_dataset = self.intersperse_datasets(training_light_curve_and_label_datasets)
        training_dataset = self.window_dataset_for_zipped_example_and_label_dataset(training_dataset, self.batch_size,
                                                                                    self.window_shift)
        validation_light_curve_and_label_datasets = []
        for index, (paths_dataset, light_curve_collection) in enumerate(
                zip(validation_standard_paths_datasets, self.validation_standard_light_curve_collections)):
            light_curve_and_label_dataset = self.generate_standard_light_curve_and_label_dataset(paths_dataset,
                                                                                                 light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                                                                                                 light_curve_collection.load_label_from_path,
                                                                                                 evaluation_mode=True,
                                                                                                 name=f"{type(light_curve_collection).__name__}_standard_validation_{index}")
            validation_light_curve_and_label_datasets.append(light_curve_and_label_dataset)
        for index, (paths_dataset, injectable_light_curve_collection) in enumerate(
                zip(validation_injectable_paths_datasets, self.validation_injectable_light_curve_collections)):
            light_curve_and_label_dataset = self.generate_injected_light_curve_and_label_dataset(
                validation_injectee_path_dataset,
                self.validation_injectee_light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                paths_dataset,
                injectable_light_curve_collection.load_times_magnifications_and_magnification_errors_from_path,
                injectable_light_curve_collection.load_label_from_path, evaluation_mode=True,
                name=f"{type(injectable_light_curve_collection).__name__}_injected_validation_{index}")
            validation_light_curve_and_label_datasets.append(light_curve_and_label_dataset)
        validation_dataset = self.intersperse_datasets(validation_light_curve_and_label_datasets)
        validation_dataset = validation_dataset.batch(self.batch_size)
        return training_dataset, validation_dataset

    def generate_paths_datasets_group_from_light_curve_collections_group(
            self, standard_light_curve_collections: List[LightCurveCollection],
            injectee_light_curve_collection: LightCurveCollection,
            injectable_light_curve_collections: List[LightCurveCollection], shuffle: bool = True
    ) -> (List[tf.data.Dataset], tf.data.Dataset, List[tf.data.Dataset]):
        """
        Create the path dataset for each light curve collection in the standard, injectee, and injectable sets.

        :param standard_light_curve_collections: The standard light curve collections.
        :param injectee_light_curve_collection: The injectee light curve collection.
        :param injectable_light_curve_collections: The injectable light curve collections.
        :param shuffle: Whether to shuffle the dataset or not.
        :return: The standard, injectee, and injectable paths datasets.
        """
        injectee_collection_index_in_standard_collection_list: Union[int, None] = None
        for index, standard_light_curve_collection in enumerate(standard_light_curve_collections):
            if standard_light_curve_collection is injectee_light_curve_collection:
                injectee_collection_index_in_standard_collection_list = index
        if injectee_collection_index_in_standard_collection_list is not None:
            standard_light_curve_collections.pop(injectee_collection_index_in_standard_collection_list)
        standard_paths_datasets = self.generate_paths_datasets_from_light_curve_collection_list(
            standard_light_curve_collections, shuffle=shuffle)
        injectee_path_dataset = None
        if injectee_light_curve_collection is not None:
            injectee_path_dataset = self.generate_paths_dataset_from_light_curve_collection(
                injectee_light_curve_collection, shuffle=shuffle)
            number_of_elements_repeated_in_a_row = len(injectable_light_curve_collections)
            if injectee_collection_index_in_standard_collection_list is not None:
                number_of_elements_repeated_in_a_row += 1
            injectee_path_dataset = injectee_path_dataset.flat_map(
                partial(repeat_each_element, number_of_repeats=number_of_elements_repeated_in_a_row))
            if injectee_collection_index_in_standard_collection_list is not None:
                standard_paths_datasets.insert(injectee_collection_index_in_standard_collection_list,
                                               injectee_path_dataset)
        injectable_paths_datasets = self.generate_paths_datasets_from_light_curve_collection_list(
            injectable_light_curve_collections, shuffle=shuffle)
        return standard_paths_datasets, injectee_path_dataset, injectable_paths_datasets

    def generate_paths_dataset_from_light_curve_collection(self, light_curve_collection: LightCurveCollection,
                                                          repeat: bool = True, shuffle: bool = True
                                                          ) -> tf.data.Dataset:
        """
        Generates a paths dataset for a light curve collection.

        :param light_curve_collection: The light curve collection to generate a paths dataset for.
        :param repeat: Whether to repeat the dataset or not.
        :param shuffle: Whether to shuffle the dataset or not.
        :return: The paths dataset.
        """
        dataset = self.paths_dataset_from_list_or_generator_factory(light_curve_collection.get_paths)
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        return dataset

    def generate_paths_datasets_from_light_curve_collection_list(self,
                                                                 light_curve_collections: List[LightCurveCollection],
                                                                 shuffle: bool = True) -> List[tf.data.Dataset]:
        """
        Generates a paths dataset for each light curve collection in a list.

        :param light_curve_collections: The list of light curve collections.
        :param shuffle: Whether to shuffle the datasets or not.
        :return: The list of paths datasets.
        """
        return [self.generate_paths_dataset_from_light_curve_collection(light_curve_collection, shuffle=shuffle)
                for light_curve_collection in light_curve_collections]

    def generate_standard_light_curve_and_label_dataset(
            self, paths_dataset: tf.data.Dataset,
            load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]], evaluation_mode: bool = False,
            name: Optional[str] = None) -> tf.data.Dataset:
        """
        Generates a light curve and label dataset from a paths dataset using a passed function defining
        how to load the values from the light curve file and the label value to use.

        :param paths_dataset: The dataset of paths to use.
        :param load_times_fluxes_and_flux_errors_from_path_function: The function defining how to load the times and
                                                                     fluxes of a light curve from a path.
        :param load_label_from_path_function: The function to load the label to use for the light curves in this dataset.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :param name: The name of the dataset.
        :return: The resulting light curve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_standard_light_curve,
                                          load_times_fluxes_and_flux_errors_from_path_function,
                                          load_label_from_path_function,
                                          evaluation_mode=evaluation_mode)
        preprocess_map_function = self.add_logging_queues_to_map_function(preprocess_map_function, name)
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, self.number_of_input_channels), (self.number_of_label_types,)]
        example_and_label_dataset = map_py_function_to_dataset(paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def add_logging_queues_to_map_function(self, preprocess_map_function: Callable, name: Optional[str]) -> Callable:
        """
        Adds logging queues to the map functions.

        :param preprocess_map_function: The function to map.
        :param name: The name of the dataset.
        :return: The updated map function.
        """
        if self.logger is not None:
            preprocess_map_function = partial(preprocess_map_function,
                                              request_queue=self.logger.create_request_queue_for_collection(name),
                                              example_queue=self.logger.create_example_queue_for_collection(name))
        return preprocess_map_function

    def preprocess_standard_light_curve(
            self,
            load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]],
            light_curve_path_tensor: tf.Tensor, evaluation_mode: bool = False,
            request_queue: Optional[Queue] = None,
            example_queue: Optional[Queue] = None
    ) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual standard light curve from a light curve path tensor, using a passed function defining
        how to load the values from the light curve file and the label value to use. Designed to be used with `partial`
        to prepare a function which will just require the light curve path tensor, and can then be mapped to a dataset.

        :param load_times_fluxes_and_flux_errors_from_path_function: The function to load the light curve times and
                                                                     fluxes from a file.
        :param load_label_from_path_function: The function to load the label to assign to the light curve.
        :param light_curve_path_tensor: The tensor containing the path to the light curve file.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :param request_queue: The logging request queue.
        :param example_queue: The logging example queue.
        :return: The example and label arrays shaped for use as single example for the network.
        """
        light_curve_path = Path(light_curve_path_tensor.numpy().decode('utf-8'))
        times, fluxes, flux_errors = load_times_fluxes_and_flux_errors_from_path_function(light_curve_path)
        if self.logger is not None and self.logger.should_produce_example(request_queue):
            light_curve = LightCurve.from_times_and_fluxes(times, fluxes)
            loggable_light_curve = WandbLoggableLightCurve(light_curve_name=light_curve_path.name,
                                                           light_curve=light_curve)
            self.logger.submit_loggable(example_queue, loggable_light_curve)
        light_curve = self.build_light_curve_array(fluxes=fluxes, times=times, flux_errors=flux_errors)
        example = self.preprocess_light_curve(light_curve, evaluation_mode=evaluation_mode)
        label = load_label_from_path_function(light_curve_path)
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

    def generate_infer_path_and_light_curve_dataset(
            self, paths_dataset: tf.data.Dataset,
            load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]]):
        """
        Generates a path and light curve dataset from a paths dataset using a passed function defining
        how to load the values from the light curve file.

        :param paths_dataset: The dataset of paths to use.
        :param load_times_fluxes_and_flux_errors_from_path_function: The function defining how to load the times and
                                                                     fluxes of a light curve from a path.
        :return: The resulting light curve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_infer_light_curve,
                                          load_times_fluxes_and_flux_errors_from_path_function)
        output_types = (tf.string, tf.float32)
        output_shapes = [(), (self.time_steps_per_example, self.number_of_input_channels)]
        example_and_label_dataset = map_py_function_to_dataset(paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_infer_light_curve(
            self, load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            light_curve_path_tensor: tf.Tensor) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual standard light curve from a light curve path tensor, using a passed function defining
        how to load the values from the light curve file and returns the path and light curve. Designed to be used with
        `partial` to prepare a function which will just require the light curve path tensor, and can then be mapped to a
        dataset.

        :param load_times_fluxes_and_flux_errors_from_path_function: The function to load the light curve times and
                                                                     fluxes from a file.
        :param light_curve_path_tensor: The tensor containing the path to the light curve file.
        :return: The path and example array shaped for use as single example for the network.
        """
        light_curve_path_string = light_curve_path_tensor.numpy().decode('utf-8')
        light_curve_path = Path(light_curve_path_string)
        times, fluxes, flux_errors = load_times_fluxes_and_flux_errors_from_path_function(light_curve_path)
        light_curve = self.build_light_curve_array(fluxes=fluxes, times=times, flux_errors=flux_errors)
        example = self.preprocess_light_curve(light_curve, evaluation_mode=True)
        return light_curve_path_string, example

    def generate_injected_light_curve_and_label_dataset(
            self, injectee_paths_dataset: tf.data.Dataset,
            injectee_load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            injectable_paths_dataset: tf.data.Dataset,
            injectable_load_times_magnifications_and_magnification_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]], evaluation_mode: bool = False,
            name: Optional[str] = None):
        """
        Generates a light curve and label dataset from an injectee and injectable paths dataset, using passed functions
        defining how to load the values from the light curve files for each and the label value to use.

        :param injectee_paths_dataset: The dataset of paths to use for the injectee light curves.
        :param injectee_load_times_fluxes_and_flux_errors_from_path_function: The function defining how to load the
            times and fluxes of an injectee light curve from a path.
        :param injectable_paths_dataset: The dataset of paths to use for the injectable light curves.
        :param injectable_load_times_magnifications_and_magnification_errors_from_path_function: The function defining
            how to load the times and magnifications of an injectable signal from a path.
        :param load_label_from_path_function: The function to load the label to use for the light curves in this dataset.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :param name: The name of the dataset.
        :return: The resulting light curve example and label dataset.
        """
        preprocess_map_function = partial(
            self.preprocess_injected_light_curve,
            injectee_load_times_fluxes_and_flux_errors_from_path_function,
            injectable_load_times_magnifications_and_magnification_errors_from_path_function,
            load_label_from_path_function,
            evaluation_mode=evaluation_mode)
        preprocess_map_function = self.add_logging_queues_to_map_function(preprocess_map_function, name)
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, self.number_of_input_channels), (self.number_of_label_types,)]
        zipped_paths_dataset = tf.data.Dataset.zip((injectee_paths_dataset, injectable_paths_dataset))
        example_and_label_dataset = map_py_function_to_dataset(zipped_paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_injected_light_curve(
            self,
            injectee_load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            injectable_load_times_magnifications_and_magnification_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]],
            injectee_light_curve_path_tensor: tf.Tensor, injectable_light_curve_path_tensor: tf.Tensor,
            evaluation_mode: bool = False, request_queue: Optional[Queue] = None,
            example_queue: Optional[Queue] = None
    ) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual injected light curve from an injectee and an injectable light curve path tensor,
        using a passed function defining how to load the values from each light curve file and the label value to use.
        Designed to be used with `partial` to prepare a function which will just require the light curve path tensor,
        and can then be mapped to a dataset.

        :param injectee_load_times_fluxes_and_flux_errors_from_path_function: The function to load the injectee
            light curve times and fluxes from a file.
        :param injectable_load_times_magnifications_and_magnification_errors_from_path_function: The function to load
            the injectee light curve times and signal from a file.
        :param load_label_from_path_function: The function to load the label to assign to the light curve.
        :param injectee_light_curve_path_tensor: The tensor containing the path to the injectee light curve file.
        :param injectable_light_curve_path_tensor: The tensor containing the path to the injectable light curve file.
        :param evaluation_mode: Whether or not the preprocessing should occur in evaluation mode (for repeatability).
        :param request_queue: The logging request queue.
        :param example_queue: The logging example queue.
        :return: The injected example and label arrays shaped for use as single example for the network.
        """
        injectee_light_curve_path = Path(injectee_light_curve_path_tensor.numpy().decode('utf-8'))
        injectee_arrays = injectee_load_times_fluxes_and_flux_errors_from_path_function(injectee_light_curve_path)
        injectee_times, injectee_fluxes, injectee_flux_errors = injectee_arrays
        injectable_light_curve_path = Path(injectable_light_curve_path_tensor.numpy().decode('utf-8'))
        injectable_arrays = injectable_load_times_magnifications_and_magnification_errors_from_path_function(
            injectable_light_curve_path)
        injectable_times, injectable_magnifications, injectable_magnification_errors = injectable_arrays
        if injectee_flux_errors is not None or injectable_magnification_errors is not None:
            raise NotImplementedError
        loggable_injection = None
        if self.logger is not None and self.logger.should_produce_example(request_queue):
            loggable_injection = WandbLoggableInjection()
        fluxes = self.inject_signal_into_light_curve(injectee_fluxes, injectee_times, injectable_magnifications,
                                                     injectable_times, loggable_injection)
        if loggable_injection is not None:
            loggable_injection.injectee_name = injectee_light_curve_path.name
            loggable_injection.injectee_light_curve = LightCurve.from_times_and_fluxes(injectee_times, injectee_fluxes)
            loggable_injection.injectable_name = injectable_light_curve_path.name
            loggable_injection.injectable_light_curve = LightCurve.from_times_and_fluxes(injectable_times,
                                                                                         injectable_magnifications)
            loggable_injection.injected_light_curve = LightCurve.from_times_and_fluxes(injectee_times, fluxes)
            self.logger.submit_loggable(example_queue=example_queue, loggable=loggable_injection)
        light_curve = self.build_light_curve_array(fluxes=fluxes, times=injectee_times)
        example = self.preprocess_light_curve(light_curve, evaluation_mode=evaluation_mode)
        label = load_label_from_path_function(injectable_light_curve_path)
        label = self.expand_label_to_training_dimensions(label)
        return example, label

    def inject_signal_into_light_curve(self, light_curve_fluxes: np.ndarray, light_curve_times: np.ndarray,
                                      signal_magnifications: np.ndarray, signal_times: np.ndarray,
                                      wandb_loggable_injection: Optional[WandbLoggableInjection] = None) -> np.ndarray:
        """
        Injects a synthetic magnification signal into real light curve fluxes.

        :param light_curve_fluxes: The fluxes of the light curve to be injected into.
        :param light_curve_times: The times of the flux observations of the light curve.
        :param signal_magnifications: The synthetic magnifications to inject.
        :param signal_times: The times of the synthetic magnifications.
        :param wandb_loggable_injection: The object to log the injection process.
        :return: The fluxes with the injected signal.
        """
        minimum_light_curve_time = np.min(light_curve_times)
        relative_light_curve_times = light_curve_times - minimum_light_curve_time
        relative_signal_times = signal_times - np.min(signal_times)
        signal_time_length = np.max(relative_signal_times)
        light_curve_time_length = np.max(relative_light_curve_times)
        time_length_difference = light_curve_time_length - signal_time_length
        signal_start_offset = (np.random.random() * time_length_difference) + minimum_light_curve_time
        offset_signal_times = relative_signal_times + signal_start_offset
        if self.baseline_flux_estimation_method == BaselineFluxEstimationMethod.MEDIAN_ABSOLUTE_DEVIATION:
            baseline_flux = scipy.stats.median_abs_deviation(light_curve_fluxes)
            baseline_to_median_absolute_deviation_ratio = 10  # Arbitrarily chosen to give a reasonable scale.
            baseline_flux *= baseline_to_median_absolute_deviation_ratio
        else:
            baseline_flux = np.median(light_curve_fluxes)
        signal_fluxes = (signal_magnifications * baseline_flux) - baseline_flux
        if self.out_of_bounds_injection_handling is OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION:
            signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=False, fill_value=0)
        elif (self.out_of_bounds_injection_handling is OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL and
              time_length_difference > 0):
            before_signal_gap = signal_start_offset - minimum_light_curve_time
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
        interpolated_signal_fluxes = signal_flux_interpolator(light_curve_times)
        fluxes_with_injected_signal = light_curve_fluxes + interpolated_signal_fluxes
        if wandb_loggable_injection is not None:
            wandb_loggable_injection.aligned_injectee_light_curve = LightCurve.from_times_and_fluxes(
                light_curve_times, light_curve_fluxes)
            wandb_loggable_injection.aligned_injectable_light_curve = LightCurve.from_times_and_fluxes(
                offset_signal_times, signal_fluxes)
            wandb_loggable_injection.aligned_injected_light_curve = LightCurve.from_times_and_fluxes(
                light_curve_times, fluxes_with_injected_signal)
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
        for light_curve_collection in self.inference_light_curve_collections:
            example_paths_dataset = self.generate_paths_dataset_from_light_curve_collection(light_curve_collection,
                                                                                            repeat=False, shuffle=False)
            examples_dataset = self.generate_infer_path_and_light_curve_dataset(
                example_paths_dataset, light_curve_collection.load_times_fluxes_and_flux_errors_from_path)
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
