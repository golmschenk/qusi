"""
Code to represent the database for injecting synthetic signals into real TESS data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase
from ramjet.photometric_database.tess_data_interface import TessDataInterface


class TessSyntheticInjectedDatabase(LightcurveDatabase):
    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/microlensing')
        self.lightcurve_directory: Path = self.data_directory.joinpath('lightcurves')
        self.synthetic_signal_directory: Path = self.data_directory.joinpath('synthetic_signals')
        self.tess_data_interface = TessDataInterface()

    def generate_datasets(self):
        all_lightcurve_paths = list(self.lightcurve_directory.glob('*.fits'))
        all_synthetic_paths = list(self.synthetic_signal_directory.glob('*.feather'))
        lightcurve_paths_datasets = self.get_training_and_validation_datasets_for_file_paths(all_lightcurve_paths)
        training_lightcurve_paths_dataset, validation_lightcurve_paths_dataset = lightcurve_paths_datasets
        shuffled_training_dataset = training_lightcurve_paths_dataset.shuffle(
            buffer_size=len(training_lightcurve_paths_dataset))
        lightcurve_training_dataset = shuffled_training_dataset.map(self.training_preprocessing, num_parallel_calls=16)
        injected_and_not_injected_lightcurve_training_dataset = shuffled_training_dataset.map(self.injection_function,
                                                                                              num_parallel_calls=16)
        batched_training_dataset = lightcurve_training_dataset.batch(100)
        prefetch_training_dataset = batched_training_dataset.prefetch(10)

    def general_preprocessing(self, lightcurve_path_tensor: tf.Tensor, synthetic_signal_path_tensor: tf.Tensor
                              ) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        lightcurve_path = lightcurve_path_tensor.numpy().decode('utf-8')
        synthetic_signal_path = synthetic_signal_path_tensor.numpy().decode('utf-8')
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path)
        synthetic_signal = pd.read_feather(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = synthetic_signal['Magnification'], synthetic_signal['Time (hours)']

        lightcurve = np.expand_dims(fluxes, axis=-1)

    @staticmethod
    def inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times, signal_magnifications, signal_times):
        median_flux = np.median(lightcurve_fluxes)
        signal_fluxes = (signal_magnifications * median_flux) - median_flux
        time_differences = np.diff(lightcurve_times, prepend=lightcurve_times[0])
        signal_flux_interpolator = interp1d(signal_times, signal_fluxes, bounds_error=True)
        lightcurve_relative_times = np.cumsum(time_differences)
        interpolated_signal_fluxes = signal_flux_interpolator(lightcurve_relative_times)
        fluxes_with_injected_signal = lightcurve_fluxes + interpolated_signal_fluxes
        return fluxes_with_injected_signal

    def training_preprocessing(self):
        pass

    def evaluation_preprocessing(self):
        pass