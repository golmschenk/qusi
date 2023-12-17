"""
An abstract class allowing for any number and combination of standard and injectable/injectee light curve collections.
"""
from functools import partial
from queue import Queue

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Callable, Tuple, Optional

from ramjet.logging.wandb_logger import WandbLogger, WandbLoggableLightCurve, \
    WandbLoggableInjection
from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.light_curve_collection import LightCurveCollection
from ramjet.photometric_database.light_curve_database import LightCurveDatabase, \
    paths_dataset_from_list_or_generator_factory
from ramjet.photometric_database.light_curve_dataset_manipulations import OutOfBoundsInjectionHandlingMethod, \
    BaselineFluxEstimationMethod, inject_signal_into_light_curve_with_intermediates


def flat_window_zipped_example_and_label_dataset(dataset: tf.data.Dataset, batch_size: int, window_shift: int,
                                                 ) -> tf.data.Dataset:
    """
    Takes a zipped example and label dataset and repeats examples in a windowed fashion of a given batch size.
    It is expected that the resulting dataset will subsequently be batched in some fashion by the given batch size.

    :param dataset: The zipped example and label dataset.
    :param batch_size: The size of the batches to produce.
    :param window_shift: The shift of the moving window between batches.
    :return: The flattened window dataset.
    """
    if window_shift != 0:
        windowed_dataset = dataset.window(batch_size, shift=window_shift)
        unbatched_window_dataset = windowed_dataset.flat_map(
            lambda *sample: tf.data.Dataset.zip(tuple(element for element in sample)))
        return unbatched_window_dataset
    else:
        return dataset


def padded_window_dataset_for_zipped_example_and_label_dataset(dataset: tf.data.Dataset, batch_size: int,
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
    unbatched_window_dataset = flat_window_zipped_example_and_label_dataset(dataset, batch_size,
                                                                            window_shift)
    return unbatched_window_dataset.padded_batch(batch_size, padded_shapes=padded_shapes)


def window_dataset_for_zipped_example_and_label_dataset(dataset: tf.data.Dataset, batch_size: int,
                                                        window_shift: int) -> tf.data.Dataset:
    """
    Takes a zipped example and label dataset, and converts it to batches, where each batch uses overlapping
    examples based on a sliding window.

    :param dataset: The zipped example and label dataset.
    :param batch_size: The size of the batches to produce.
    :param window_shift: The shift of the moving window between batches.
    :return: The window dataset.
    """
    unbatched_window_dataset = flat_window_zipped_example_and_label_dataset(dataset, batch_size,
                                                                            window_shift)
    return unbatched_window_dataset.batch(batch_size)


def inject_signal_into_light_curve(
        light_curve_times: npt.NDArray[np.float64],
        light_curve_fluxes: npt.NDArray[np.float64],
        signal_times: npt.NDArray[np.float64],
        signal_magnifications: npt.NDArray[np.float64],
        out_of_bounds_injection_handling_method: OutOfBoundsInjectionHandlingMethod =
        OutOfBoundsInjectionHandlingMethod.ERROR,
        baseline_flux_estimation_method: BaselineFluxEstimationMethod = BaselineFluxEstimationMethod.MEDIAN
) -> npt.NDArray[np.float64]:
    """
    Injects a synthetic magnification signal into real light curve fluxes.

    :param light_curve_times: The times of the flux observations of the light curve.
    :param light_curve_fluxes: The fluxes of the light curve to be injected into.
    :param signal_times: The times of the synthetic magnifications.
    :param signal_magnifications: The synthetic magnifications to inject.
    :param out_of_bounds_injection_handling_method: The method to use to handle out of bounds injection.
    :param baseline_flux_estimation_method: The method to use to estimate the baseline flux of the light curve
                                            for scaling the signal magnifications.
    :return: The fluxes with the injected signal, the offset signal times, and the signal flux.
    """
    fluxes_with_injected_signal, _, _ = inject_signal_into_light_curve_with_intermediates(
        light_curve_times=light_curve_times,
        light_curve_fluxes=light_curve_fluxes,
        signal_times=signal_times,
        signal_magnifications=signal_magnifications,
        out_of_bounds_injection_handling_method=out_of_bounds_injection_handling_method,
        baseline_flux_estimation_method=baseline_flux_estimation_method
    )
    return fluxes_with_injected_signal


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
        self.number_of_label_values = 1
        self.number_of_auxiliary_values: int = 0
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
        dataset = paths_dataset_from_list_or_generator_factory(light_curve_collection.get_paths)
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
            load_auxiliary_information_for_path_function: Callable[[Path], np.ndarray],
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
        label = expand_label_to_training_dimensions(label)
        if self.number_of_auxiliary_values > 0:
            auxiliary_information = load_auxiliary_information_for_path_function(light_curve_path)
            return example, auxiliary_information, label
        else:
            return example, label

    def preprocess_infer_light_curve(
            self, load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_auxiliary_information_for_path_function: Callable[[Path], np.ndarray],
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
        if self.number_of_auxiliary_values > 0:
            auxiliary_information = load_auxiliary_information_for_path_function(light_curve_path)
            return light_curve_path_string, example, auxiliary_information
        else:
            return light_curve_path_string, example

    def preprocess_injected_light_curve(
            self,
            injectee_load_times_fluxes_and_flux_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_auxiliary_information_for_path_function: Callable[[Path], np.ndarray],
            injectable_load_times_magnifications_and_magnification_errors_from_path_function: Callable[
                [Path], Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]],
            load_label_from_path_function: Callable[[Path], Union[float, np.ndarray]],
            injectee_light_curve_path_tensor: tf.Tensor, injectable_light_curve_path_tensor: tf.Tensor,
            evaluation_mode: bool = False, request_queue: Optional[Queue] = None,
            example_queue: Optional[Queue] = None
    ) -> (np.ndarray, np.ndarray):
        """
        Preprocesses an individual injected light curve from an injectee and an injectable light curve path tensor,
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
        label = expand_label_to_training_dimensions(label)
        if self.number_of_auxiliary_values > 0:
            auxiliary_information = load_auxiliary_information_for_path_function(injectee_light_curve_path)
            return example, auxiliary_information, label
        else:
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
        out_of_bounds_injection_handling_method = self.out_of_bounds_injection_handling
        baseline_flux_estimation_method = self.baseline_flux_estimation_method
        fluxes_with_injected_signal, offset_signal_times, signal_fluxes = inject_signal_into_light_curve_with_intermediates(
            light_curve_times, light_curve_fluxes, signal_times, signal_magnifications,
            out_of_bounds_injection_handling_method, baseline_flux_estimation_method)
        if wandb_loggable_injection is not None:
            wandb_loggable_injection.aligned_injectee_light_curve = LightCurve.from_times_and_fluxes(
                light_curve_times, light_curve_fluxes)
            wandb_loggable_injection.aligned_injectable_light_curve = LightCurve.from_times_and_fluxes(
                offset_signal_times, signal_fluxes)
            wandb_loggable_injection.aligned_injected_light_curve = LightCurve.from_times_and_fluxes(
                light_curve_times, fluxes_with_injected_signal)
        return fluxes_with_injected_signal


def from_path_light_curve_and_auxiliary_to_path_and_observation(
        light_curve_auxiliary_and_label_dataset: tf.data.Dataset) -> tf.data.Dataset:
    path_observation_dataset = light_curve_auxiliary_and_label_dataset.map(
        lambda path, light_curve, auxiliary: (path, (light_curve, auxiliary)))
    return path_observation_dataset


def from_light_curve_auxiliary_and_label_to_observation_and_label(
        light_curve_auxiliary_and_label_dataset: tf.data.Dataset) -> tf.data.Dataset:
    observation_and_label_dataset = light_curve_auxiliary_and_label_dataset.map(
        lambda light_curve, auxiliary, label: ((light_curve, auxiliary), label))
    return observation_and_label_dataset


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


def repeat_each_element(element: tf.Tensor, number_of_repeats: int) -> tf.data.Dataset:
    """
    A dataset mappable function which repeats the elements a given number of times.

    :param element: The element to map to repeat.
    :param number_of_repeats: The number of times to repeat the element.
    :return: The dataset with repeated elements.
    """
    return tf.data.Dataset.from_tensors(element).repeat(number_of_repeats)
