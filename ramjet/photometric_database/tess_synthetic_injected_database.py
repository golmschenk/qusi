"""
Code to represent the database for injecting synthetic signals into real TESS data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase
from ramjet.photometric_database.py_mapper import map_py_function_to_dataset
from ramjet.photometric_database.tess_data_interface import TessDataInterface


class TessSyntheticInjectedDatabase(LightcurveDatabase):
    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/microlensing')
        self.lightcurve_directory: Path = self.data_directory.joinpath('lightcurves')
        self.synthetic_signal_directory: Path = self.data_directory.joinpath('synthetic_signals')
        self.tess_data_interface = TessDataInterface()
        self.time_steps_per_example = 20000

    def generate_datasets(self):
        all_lightcurve_paths = list(self.lightcurve_directory.glob('*.fits'))
        all_synthetic_paths = list(map(str, self.synthetic_signal_directory.glob('*.feather')))
        synthetic_signal_paths_dataset = tf.data.Dataset.from_tensor_slices(all_synthetic_paths)
        lightcurve_paths_datasets = self.get_training_and_validation_datasets_for_file_paths(all_lightcurve_paths)
        training_lightcurve_paths_dataset, validation_lightcurve_paths_dataset = lightcurve_paths_datasets
        shuffled_training_lightcurve_paths_dataset = training_lightcurve_paths_dataset.shuffle(
            buffer_size=len(list(training_lightcurve_paths_dataset)))
        shuffled_synthetic_signal_paths_dataset = synthetic_signal_paths_dataset.shuffle(
            buffer_size=len(list(synthetic_signal_paths_dataset)))
        zipped_training_paths_dataset = tf.data.Dataset.zip((shuffled_training_lightcurve_paths_dataset,
                                                             shuffled_synthetic_signal_paths_dataset))
        output_types = ((tf.float32, tf.float32), (tf.float32, tf.float32))
        lightcurve_training_dataset = map_py_function_to_dataset(zipped_training_paths_dataset,
                                                                 self.train_and_validation_preprocessing,
                                                                 self.number_of_parallel_processes_per_map,
                                                                 output_types=output_types,
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
                                                                   flat_map=True)
        batched_validation_dataset = lightcurve_validation_dataset.batch(self.batch_size)
        prefetch_validation_dataset = batched_validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return prefetch_training_dataset, prefetch_validation_dataset

    def train_and_validation_preprocessing(self, lightcurve_path_tensor: tf.Tensor,
                                           synthetic_signal_path_tensor: tf.Tensor,
                                           ) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        """
        The training and validation preprocessing.

        :param lightcurve_path_tensor: The lightcurve's path to be preprocessed.
        :param synthetic_signal_path_tensor: The synthetic signal's path to be injected.
        :return: Two examples, one negative uninjected signal and one positive injected signal, both with labels.
        """
        lightcurve_path = lightcurve_path_tensor.numpy().decode('utf-8')
        synthetic_signal_path = synthetic_signal_path_tensor.numpy().decode('utf-8')
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path)
        synthetic_signal = pd.read_feather(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = synthetic_signal['Magnification'], synthetic_signal['Time (hours)']
        synthetic_times = synthetic_times / 24  # Convert hours to days.
        fluxes_with_injected_signal = self.inject_signal_into_lightcurve(fluxes, times, synthetic_magnifications,
                                                                         synthetic_times)
        fluxes = self.flux_preprocessing(fluxes)
        fluxes_with_injected_signal = self.flux_preprocessing(fluxes_with_injected_signal)
        lightcurve = np.expand_dims(fluxes, axis=-1)
        lightcurve_with_injected_signal = np.expand_dims(fluxes_with_injected_signal, axis=-1)
        return (lightcurve, np.array([0])), (lightcurve_with_injected_signal, np.array([1]))

    def flux_preprocessing(self, fluxes: np.ndarray, evaluation_mode=False) -> np.ndarray:
        normalized_fluxes = self.normalize(fluxes)
        uniform_length_fluxes = self.make_uniform_length(normalized_fluxes, self.time_steps_per_example,
                                                         randomize=not evaluation_mode)
        return uniform_length_fluxes

    @staticmethod
    def inject_signal_into_lightcurve(lightcurve_fluxes: np.ndarray, lightcurve_times: np.ndarray,
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
        time_differences = np.diff(lightcurve_times, prepend=lightcurve_times[0])
        signal_flux_interpolator = interp1d(signal_times, signal_fluxes, bounds_error=True)
        lightcurve_relative_times = np.cumsum(time_differences)
        interpolated_signal_fluxes = signal_flux_interpolator(lightcurve_relative_times)
        fluxes_with_injected_signal = lightcurve_fluxes + interpolated_signal_fluxes
        return fluxes_with_injected_signal
