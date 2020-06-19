"""
An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections.
"""
from functools import partial

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Callable, Tuple

from scipy.interpolate import interp1d

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection
from ramjet.photometric_database.lightcurve_database import LightcurveDatabase
from ramjet.py_mapper import map_py_function_to_dataset


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
        self.shuffle_buffer_size = 10000
        self.time_steps_per_example = 20000
        self.allow_out_of_bounds_injection = False

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets for the database.

        :return: The training and validation dataset.
        """
        training_standard_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.training_standard_lightcurve_collections)
        training_injectee_path_dataset = None
        if self.training_injectee_lightcurve_collection is not None:
            training_injectee_path_dataset = self.generate_paths_dataset_from_lightcurve_collection(
                self.training_injectee_lightcurve_collection)
        training_injectable_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.training_injectable_lightcurve_collections)
        validation_standard_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.validation_standard_lightcurve_collections)
        validation_injectee_path_dataset = None
        if self.validation_injectee_lightcurve_collection is not None:
            validation_injectee_path_dataset = self.generate_paths_dataset_from_lightcurve_collection(
                self.validation_injectee_lightcurve_collection)
        validation_injectable_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.validation_injectable_lightcurve_collections)
        training_lightcurve_and_label_datasets = []
        for paths_dataset, lightcurve_collection in zip(training_standard_paths_datasets,
                                                        self.training_standard_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_standard_lightcurve_and_label_dataset(
                paths_dataset, lightcurve_collection.load_times_and_fluxes_from_path,
                lightcurve_collection.label
            )
            training_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        for paths_dataset, injectable_lightcurve_collection in zip(training_injectable_paths_datasets,
                                                                   self.training_injectable_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_injected_lightcurve_and_label_dataset(
                training_injectee_path_dataset,
                self.training_injectee_lightcurve_collection.load_times_and_fluxes_from_path,
                paths_dataset, injectable_lightcurve_collection.load_times_and_magnifications_from_path,
                injectable_lightcurve_collection.label
            )
            training_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        training_dataset = self.intersperse_datasets(training_lightcurve_and_label_datasets)
        training_dataset = self.window_dataset_for_zipped_example_and_label_dataset(training_dataset, self.batch_size,
                                                                                    self.batch_size // 10)
        validation_lightcurve_and_label_datasets = []
        for paths_dataset, lightcurve_collection in zip(validation_standard_paths_datasets,
                                                        self.validation_standard_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_standard_lightcurve_and_label_dataset(
                paths_dataset, lightcurve_collection.load_times_and_fluxes_from_path,
                lightcurve_collection.label
            )
            validation_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        for paths_dataset, injectable_lightcurve_collection in zip(validation_injectable_paths_datasets,
                                                                   self.validation_injectable_lightcurve_collections):
            lightcurve_and_label_dataset = self.generate_injected_lightcurve_and_label_dataset(
                validation_injectee_path_dataset,
                self.validation_injectee_lightcurve_collection.load_times_and_fluxes_from_path,
                paths_dataset, injectable_lightcurve_collection.load_times_and_magnifications_from_path,
                injectable_lightcurve_collection.label
            )
            validation_lightcurve_and_label_datasets.append(lightcurve_and_label_dataset)
        validation_dataset = self.intersperse_datasets(validation_lightcurve_and_label_datasets)
        validation_dataset = self.window_dataset_for_zipped_example_and_label_dataset(validation_dataset,
                                                                                      self.batch_size,
                                                                                      self.batch_size // 10)
        return training_dataset, validation_dataset

    def generate_paths_dataset_from_lightcurve_collection(self, lightcurve_collection: LightcurveCollection
                                                          ) -> tf.data.Dataset:
        """
        Generates a paths dataset for a lightcurve collection.

        :param lightcurve_collection: The lightcurve collection to generate a paths dataset for.
        :return: The paths dataset.
        """
        paths_dataset = self.paths_dataset_from_list_or_generator_factory(lightcurve_collection.get_paths)
        repeated_paths_dataset = paths_dataset.repeat()
        shuffled_paths_dataset = repeated_paths_dataset.shuffle(self.shuffle_buffer_size)
        return shuffled_paths_dataset

    def generate_paths_datasets_from_lightcurve_collection_list(self, lightcurve_collections: List[LightcurveCollection]
                                                                ) -> List[tf.data.Dataset]:
        """
        Generates a paths dataset for each lightcurve collection in a list.

        :param lightcurve_collections: The list of lightcurve collections.
        :return: The list of paths datasets.
        """
        return [self.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
                for lightcurve_collection in lightcurve_collections]

    def generate_standard_lightcurve_and_label_dataset(
            self, paths_dataset: tf.data.Dataset,
            load_times_and_fluxes_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]], label: float):
        """
        Generates a lightcurve and label dataset from a paths dataset using a passed function defining
        how to load the values from the lightcurve file and the label value to use.

        :param paths_dataset: The dataset of paths to use.
        :param load_times_and_fluxes_from_path_function: The function defining how to load the times and fluxes of a
                                                         lightcurve from a path.
        :param label: The label to use for the lightcurves in this dataset.
        :return: The resulting lightcurve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_standard_lightcurve, load_times_and_fluxes_from_path_function,
                                          label)
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (1,)]
        example_and_label_dataset = map_py_function_to_dataset(paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_standard_lightcurve(
            self, load_times_and_fluxes_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]],
            label: float, lightcurve_path_tensor: tf.Tensor) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual standard lightcurve from a lightcurve path tensor, using a passed function defining
        how to load the values from the lightcurve file and the label value to use. Designed to be used with `partial`
        to prepare a function which will just require the lightcurve path tensor, and can then be mapped to a dataset.

        :param load_times_and_fluxes_from_path_function: The function to load the lightcurve times and fluxes from a
                                                         file.
        :param label: The label to assign to the lightcurve.
        :param lightcurve_path_tensor: The tensor containing the path to the lightcurve file.
        :return: The example and label arrays shaped for use as single example for the network.
        """
        lightcurve_path = Path(lightcurve_path_tensor.numpy().decode('utf-8'))
        times, fluxes = load_times_and_fluxes_from_path_function(lightcurve_path)
        preprocessed_fluxes = self.flux_preprocessing(fluxes)
        example = np.expand_dims(preprocessed_fluxes, axis=-1)
        return example, np.array([label])

    def generate_infer_path_and_lightcurve_dataset(
            self, paths_dataset: tf.data.Dataset,
            load_times_and_fluxes_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]]):
        """
        Generates a path and lightcurve dataset from a paths dataset using a passed function defining
        how to load the values from the lightcurve file.

        :param paths_dataset: The dataset of paths to use.
        :param load_times_and_fluxes_from_path_function: The function defining how to load the times and fluxes of a
                                                         lightcurve from a path.
        :return: The resulting lightcurve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_infer_lightcurve, load_times_and_fluxes_from_path_function)
        output_types = (tf.string, tf.float32)
        output_shapes = [(), (self.time_steps_per_example, 1)]
        example_and_label_dataset = map_py_function_to_dataset(paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_infer_lightcurve(
            self, load_times_and_fluxes_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]],
            lightcurve_path_tensor: tf.Tensor) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual standard lightcurve from a lightcurve path tensor, using a passed function defining
        how to load the values from the lightcurve file and returns the path and lightcurve. Designed to be used with
        `partial` to prepare a function which will just require the lightcurve path tensor, and can then be mapped to a
        dataset.

        :param load_times_and_fluxes_from_path_function: The function to load the lightcurve times and fluxes from a
                                                         file.
        :param lightcurve_path_tensor: The tensor containing the path to the lightcurve file.
        :return: The path and example array shaped for use as single example for the network.
        """
        lightcurve_path_string = lightcurve_path_tensor.numpy().decode('utf-8')
        lightcurve_path = Path(lightcurve_path_string)
        times, fluxes = load_times_and_fluxes_from_path_function(lightcurve_path)
        preprocessed_fluxes = self.flux_preprocessing(fluxes)
        example = np.expand_dims(preprocessed_fluxes, axis=-1)
        return lightcurve_path_string, example

    def generate_injected_lightcurve_and_label_dataset(
            self, injectee_paths_dataset: tf.data.Dataset,
            injectee_load_times_and_fluxes_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]],
            injectable_paths_dataset: tf.data.Dataset,
            injectable_load_times_and_magnifications_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]],
            label: float):
        """
        Generates a lightcurve and label dataset from an injectee and injectable paths dataset, using passed functions
        defining how to load the values from the lightcurve files for each and the label value to use.

        :param injectee_paths_dataset: The dataset of paths to use for the injectee lightcurves.
        :param injectee_load_times_and_fluxes_from_path_function: The function defining how to load the times and fluxes
                                                                  of an injectee lightcurve from a path.
        :param injectable_paths_dataset: The dataset of paths to use for the injectable lightcurves.
        :param injectable_load_times_and_magnifications_from_path_function: The function defining how to load the times
                                                                            and magnifications of an injectable
                                                                            signal from a path.
        :param label: The label to use for the lightcurves in this dataset.
        :return: The resulting lightcurve example and label dataset.
        """
        preprocess_map_function = partial(self.preprocess_injected_lightcurve,
                                          injectee_load_times_and_fluxes_from_path_function,
                                          injectable_load_times_and_magnifications_from_path_function,
                                          label)
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (1,)]
        zipped_paths_dataset = tf.data.Dataset.zip((injectee_paths_dataset, injectable_paths_dataset))
        example_and_label_dataset = map_py_function_to_dataset(zipped_paths_dataset,
                                                               preprocess_map_function,
                                                               self.number_of_parallel_processes_per_map,
                                                               output_types=output_types,
                                                               output_shapes=output_shapes)
        return example_and_label_dataset

    def preprocess_injected_lightcurve(
            self, injectee_load_times_and_fluxes_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]],
            injectable_load_times_and_magnifications_from_path_function: Callable[[Path], Tuple[np.ndarray, np.ndarray]],
            label: float, injectee_lightcurve_path_tensor: tf.Tensor, injectable_lightcurve_path_tensor: tf.Tensor
            ) -> (np.ndarray, np.ndarray):
        """
        Preprocesses a individual injected lightcurve from an injectee and an injectable lightcurve path tensor,
        using a passed function defining how to load the values from each lightcurve file and the label value to use.
        Designed to be used with `partial` to prepare a function which will just require the lightcurve path tensor, and
        can then be mapped to a dataset.

        :param injectee_load_times_and_fluxes_from_path_function: The function to load the injectee lightcurve times and
                                                                  fluxes from a file.
        :param injectable_load_times_and_magnifications_from_path_function: The function to load the injectee lightcurve
                                                                            times and signal from a file.
        :param label: The label to assign to the lightcurve.
        :param injectee_lightcurve_path_tensor: The tensor containing the path to the injectee lightcurve file.
        :param injectable_lightcurve_path_tensor: The tensor containing the path to the injectable lightcurve file.
        :return: The injected example and label arrays shaped for use as single example for the network.
        """
        injectee_lightcurve_path = Path(injectee_lightcurve_path_tensor.numpy().decode('utf-8'))
        injectee_times, injectee_fluxes = injectee_load_times_and_fluxes_from_path_function(injectee_lightcurve_path)
        injectable_lightcurve_path = Path(injectable_lightcurve_path_tensor.numpy().decode('utf-8'))
        injectable_times, injectable_magnifications = injectable_load_times_and_magnifications_from_path_function(
            injectable_lightcurve_path)
        fluxes = self.inject_signal_into_lightcurve(injectee_fluxes, injectee_times, injectable_magnifications,
                                                    injectable_times)
        preprocessed_fluxes = self.flux_preprocessing(fluxes)
        example = np.expand_dims(preprocessed_fluxes, axis=-1)
        return example, np.array([label])

    def flux_preprocessing(self, fluxes: np.ndarray, evaluation_mode: bool = False, seed: int = None) -> np.ndarray:
        """
        Preprocessing for the flux.

        :param fluxes: The flux array to preprocess.
        :param evaluation_mode: If the preprocessing should be consistent for evaluation.
        :param seed: Seed for the randomization.
        :return: The preprocessed flux array.
        """
        normalized_fluxes = self.normalize(fluxes)
        uniform_length_fluxes = self.make_uniform_length(normalized_fluxes, self.time_steps_per_example,
                                                         randomize=not evaluation_mode, seed=seed)
        return uniform_length_fluxes

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
        median_flux = np.median(lightcurve_fluxes)
        signal_fluxes = (signal_magnifications * median_flux) - median_flux
        if self.allow_out_of_bounds_injection:
            signal_flux_interpolator = interp1d(signal_times, signal_fluxes, bounds_error=False, fill_value=0)
        else:
            signal_flux_interpolator = interp1d(signal_times, signal_fluxes, bounds_error=True)
        lightcurve_relative_times = lightcurve_times - np.min(lightcurve_times)
        interpolated_signal_fluxes = signal_flux_interpolator(lightcurve_relative_times)
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
