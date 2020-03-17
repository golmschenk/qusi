"""
Code to represent the database for injecting synthetic signals into real TESS data.
"""
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase
from ramjet.photometric_database.py_mapper import map_py_function_to_dataset
from ramjet.photometric_database.tess_data_interface import TessDataInterface


class TessSyntheticInjectedDatabase(LightcurveDatabase):
    """
    A class to represent the database for injecting synthetic signals into real TESS data.
    """

    def __init__(self, data_directory='data/self_lensing_binaries'):
        super().__init__(data_directory=data_directory)
        self.lightcurve_directory: Path = self.data_directory.joinpath('lightcurves')
        self.synthetic_signal_directory: Path = self.data_directory.joinpath('synthetic_signals')
        self.tess_data_interface = TessDataInterface()
        self.time_steps_per_example = 20000
        self.allow_out_of_bounds_injection = False

    def generate_datasets(self):
        all_lightcurve_paths = self.get_all_lightcurve_paths()
        all_synthetic_paths = self.get_all_synthetic_signal_paths()
        synthetic_signal_paths_dataset = tf.data.Dataset.from_tensor_slices(all_synthetic_paths)
        lightcurve_paths_datasets = self.get_training_and_validation_datasets_for_file_paths(all_lightcurve_paths)
        training_lightcurve_paths_dataset, validation_lightcurve_paths_dataset = lightcurve_paths_datasets
        print(f'{len(list(training_lightcurve_paths_dataset))} training lightcurves.')
        print(f'{len(list(validation_lightcurve_paths_dataset))} validation lightcurves.')
        print(f'{len(list(synthetic_signal_paths_dataset))} injectable signals.')
        shuffled_training_lightcurve_paths_dataset = training_lightcurve_paths_dataset.shuffle(
            buffer_size=len(list(training_lightcurve_paths_dataset)))
        shuffled_synthetic_signal_paths_dataset = synthetic_signal_paths_dataset.shuffle(
            buffer_size=len(list(synthetic_signal_paths_dataset))).repeat()
        zipped_training_paths_dataset = tf.data.Dataset.zip((shuffled_training_lightcurve_paths_dataset,
                                                             shuffled_synthetic_signal_paths_dataset))
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (1,)]
        lightcurve_training_dataset = map_py_function_to_dataset(zipped_training_paths_dataset,
                                                                 self.train_and_validation_preprocessing,
                                                                 self.number_of_parallel_processes_per_map,
                                                                 output_types=output_types,
                                                                 output_shapes=output_shapes,
                                                                 flat_map=True)
        batched_training_dataset = lightcurve_training_dataset.batch(self.batch_size)
        prefetch_training_dataset = batched_training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        shuffled_validation_lightcurve_paths_dataset = validation_lightcurve_paths_dataset.shuffle(
            buffer_size=len(list(validation_lightcurve_paths_dataset)))
        zipped_validation_paths_dataset = tf.data.Dataset.zip((shuffled_validation_lightcurve_paths_dataset,
                                                               shuffled_synthetic_signal_paths_dataset))
        lightcurve_validation_dataset = map_py_function_to_dataset(zipped_validation_paths_dataset,
                                                                   self.train_and_validation_preprocessing,
                                                                   self.number_of_parallel_processes_per_map,
                                                                   output_types=output_types,
                                                                   output_shapes=output_shapes,
                                                                   flat_map=True)
        batched_validation_dataset = lightcurve_validation_dataset.batch(self.batch_size)
        prefetch_validation_dataset = batched_validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return prefetch_training_dataset, prefetch_validation_dataset

    def get_all_lightcurve_paths(self) -> List[str]:
        """
        Returns the list of all lightcurves to use. Expected to be overridden for subclass databases.

        :return: The list of lightcurves.
        """
        lightcurve_paths = list(map(str, self.lightcurve_directory.glob('**/*.fits')))
        return lightcurve_paths

    def get_all_synthetic_signal_paths(self) -> List[str]:
        """
        Returns the list of all synthetic signals to use. Expected to be overridden for subclass databases.

        :return: The list of synthetic signals.
        """
        synthetic_signal_paths = list(map(str, self.synthetic_signal_directory.glob('**/*.feather')))
        return synthetic_signal_paths

    def train_and_validation_preprocessing(self, lightcurve_path_tensor: tf.Tensor,
                                           synthetic_signal_path_tensor: tf.Tensor,
                                           ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        The training and validation preprocessing.

        :param lightcurve_path_tensor: The lightcurve's path to be preprocessed.
        :param synthetic_signal_path_tensor: The synthetic signal's path to be injected.
        :return: Two examples, one negative un-injected signal and one positive injected signal (paired as a tuple),
                 and the corresponding labels (paired as a tuple). Expected to have a post flat mapping to make each
                 element of the data be an individual example and label pair.
        """
        lightcurve_path = lightcurve_path_tensor.numpy().decode('utf-8')
        synthetic_signal_path = synthetic_signal_path_tensor.numpy().decode('utf-8')
        fluxes, times = self.load_fluxes_and_times_from_lightcurve_path(lightcurve_path)
        synthetic_magnifications, synthetic_times = self.load_magnifications_and_times_from_synthetic_signal_path(
            synthetic_signal_path)
        fluxes_with_injected_signal = self.inject_signal_into_lightcurve(fluxes, times, synthetic_magnifications,
                                                                         synthetic_times)
        time_seed = int(time.time())
        fluxes = self.flux_preprocessing(fluxes, seed=time_seed)
        fluxes_with_injected_signal = self.flux_preprocessing(fluxes_with_injected_signal, seed=time_seed)
        lightcurve = np.expand_dims(fluxes, axis=-1)
        lightcurve_with_injected_signal = np.expand_dims(fluxes_with_injected_signal, axis=-1)
        examples = (lightcurve, lightcurve_with_injected_signal)
        labels = (np.array([0]), np.array([1]))
        return examples, labels

    def load_fluxes_and_times_from_lightcurve_path(self, lightcurve_path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the lightcurve from the path given. Should be overridden to fit a specific database's file format.

        :param lightcurve_path: The path to the lightcurve file.
        :return: The fluxes and times of the lightcurve
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path)
        return fluxes, times

    def load_magnifications_and_times_from_synthetic_signal_path(self, synthetic_signal_path: str
                                                                 ) -> (np.ndarray, np.ndarray):
        """
        Loads the synthetic signal from the path given. Should be overridden to fit a specific database's file format.

        :param synthetic_signal_path: The path to the synthetic signal data file.
        :return: The magnifications and relative times of the synthetic signal.
        """
        synthetic_signal = pd.read_feather(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = synthetic_signal['Magnification'], synthetic_signal['Time (hours)']
        synthetic_times = synthetic_times / 24  # Convert hours to days.
        return synthetic_magnifications, synthetic_times

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
