import time
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import tensorflow as tf

from ramjet.data_interface.tess_data_interface import TessFluxType
from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase
from ramjet.py_mapper import map_py_function_to_dataset


class InjectedWithAdditionalExplicitNegativeDatabase(TessSyntheticInjectedDatabase):
    def __init__(self, data_directory='data/self_lensing_binaries'):
        super().__init__(data_directory=data_directory)
        self.explicit_negatives_csv_path = self.data_directory.joinpath('explicit_negatives.csv')

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets for the database.

        :return: The training and validation dataset.
        """
        synthetic_signal_paths_dataset = self.paths_dataset_from_list_or_generator_factory(
            self.get_all_synthetic_signal_paths)
        lightcurve_paths_datasets = self.get_training_and_validation_datasets_for_file_paths(
            self.get_all_lightcurve_paths)
        training_lightcurve_paths_dataset, validation_lightcurve_paths_dataset = lightcurve_paths_datasets
        explicit_negative_lightcurve_paths = self.paths_dataset_from_list_or_generator_factory(
            self.get_explicit_negative_lightcurve_paths)
        shuffled_training_lightcurve_paths_dataset = training_lightcurve_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_synthetic_signal_paths_dataset = synthetic_signal_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_explicit_negative_lightcurve_paths_dataset = explicit_negative_lightcurve_paths.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        zipped_training_paths_dataset = tf.data.Dataset.zip((shuffled_training_lightcurve_paths_dataset,
                                                             shuffled_synthetic_signal_paths_dataset,
                                                             shuffled_explicit_negative_lightcurve_paths_dataset))
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (1,)]
        lightcurve_training_dataset = map_py_function_to_dataset(zipped_training_paths_dataset,
                                                                 self.positive_injection_negative_and_explicit_negative_preprocessing,
                                                                 self.number_of_parallel_processes_per_map,
                                                                 output_types=output_types,
                                                                 output_shapes=output_shapes,
                                                                 flat_map=True)
        batched_training_dataset = self.window_dataset_for_zipped_example_and_label_dataset(lightcurve_training_dataset,
                                                                                            self.batch_size,
                                                                                            self.batch_size // 10)
        prefetch_training_dataset = batched_training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        shuffled_validation_lightcurve_paths_dataset = validation_lightcurve_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        zipped_validation_paths_dataset = tf.data.Dataset.zip((shuffled_validation_lightcurve_paths_dataset,
                                                               shuffled_synthetic_signal_paths_dataset,
                                                               shuffled_explicit_negative_lightcurve_paths_dataset))
        lightcurve_validation_dataset = map_py_function_to_dataset(zipped_validation_paths_dataset,
                                                                   self.positive_injection_negative_and_explicit_negative_preprocessing,
                                                                   self.number_of_parallel_processes_per_map,
                                                                   output_types=output_types,
                                                                   output_shapes=output_shapes,
                                                                   flat_map=True)
        batched_validation_dataset = lightcurve_validation_dataset.batch(self.batch_size)
        prefetch_validation_dataset = batched_validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return prefetch_training_dataset, prefetch_validation_dataset

    def positive_injection_negative_and_explicit_negative_preprocessing(
            self, lightcurve_path_tensor: tf.Tensor,
            synthetic_signal_path_tensor: tf.Tensor,
            explicit_negative_lightcurve_path_tensor: tf.Tensor
    ) -> ((np.ndarray, np.ndarray, np.ndarray), (np.ndarray, np.ndarray, np.ndarray)):
        lightcurve_path = lightcurve_path_tensor.numpy().decode('utf-8')
        synthetic_signal_path = synthetic_signal_path_tensor.numpy().decode('utf-8')
        explicit_negative_lightcurve_path = explicit_negative_lightcurve_path_tensor.numpy().decode('utf-8')
        fluxes, times = self.load_fluxes_and_times_from_lightcurve_path(lightcurve_path)
        explicit_negative_fluxes, explicit_negative_times = self.load_fluxes_and_times_from_lightcurve_path(
            explicit_negative_lightcurve_path)
        synthetic_signal = self.load_magnifications_and_times_from_synthetic_signal_path(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = synthetic_signal
        fluxes_with_injected_signal = self.inject_signal_into_lightcurve(
            fluxes, times, synthetic_magnifications, synthetic_times)
        time_seed = int(time.time())
        fluxes = self.flux_preprocessing(fluxes, seed=time_seed)
        explicit_negative_fluxes = self.flux_preprocessing(explicit_negative_fluxes, seed=time_seed)
        fluxes_with_injected_signal = self.flux_preprocessing(fluxes_with_injected_signal, seed=time_seed)
        lightcurve = np.expand_dims(fluxes, axis=-1)
        explicit_negative_lightcurve = np.expand_dims(explicit_negative_fluxes, axis=-1)
        lightcurve_with_injected_signal = np.expand_dims(fluxes_with_injected_signal, axis=-1)
        examples = (lightcurve, lightcurve_with_injected_signal, explicit_negative_lightcurve)
        labels = (np.array([0]), np.array([1]), np.array([0]))
        return examples, labels

    def get_explicit_negative_lightcurve_paths(self) -> Iterable[Path]:
        explicit_negative_lightcurve_data_frame = pd.read_csv(self.explicit_negatives_csv_path)
        explicit_negative_lightcurve_paths = explicit_negative_lightcurve_data_frame['Lightcurve path'].values
        return explicit_negative_lightcurve_paths

    def load_fluxes_and_times_from_lightcurve_path(self, lightcurve_path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the lightcurve from the path given. Should be overridden to fit a specific database's file format.

        :param lightcurve_path: The path to the lightcurve file.
        :return: The fluxes and times of the lightcurve
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path, TessFluxType.SAP)
        return fluxes, times
