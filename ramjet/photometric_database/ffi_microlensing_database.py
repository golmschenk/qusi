"""
Code to represent a database to find microlensing events in FFI data.
"""

import numpy as np
import pandas as pd
from typing import List

from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface
from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase


class FfiMicrolensingDatabase(TessSyntheticInjectedDatabase):
    """
    A class to represent a database to find microlensing events in FFI data.
    """

    def __init__(self, data_directory='data/microlensing'):
        super().__init__(data_directory=data_directory)
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.
        self.batch_size = 1000
        self.tess_ffi_data_interface = TessFfiDataInterface()

    def load_magnifications_and_times_from_synthetic_signal_path(self, synthetic_signal_path: str
                                                                 ) -> (np.ndarray, np.ndarray):
        """
        Loads the synthetic signal from the path given.

        :param synthetic_signal_path: The path to the synthetic signal data file.
        :return: The magnifications and relative times of the synthetic signal.
        """
        synthetic_signal = pd.read_feather(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = synthetic_signal['Magnification'], synthetic_signal['Time']
        synthetic_times -= 20
        # synthetic_times += np.random.random() * 30  # Synthetic data goes from -30 to 30.
        return synthetic_magnifications, synthetic_times

    def get_all_lightcurve_paths(self) -> List[str]:
        """
        Returns the list of all lightcurves to use.

        :return: The list of lightcurves.
        """
        lightcurve_paths = list(map(str, self.lightcurve_directory.glob('**/*.pkl')))
        return lightcurve_paths

    def load_fluxes_and_times_from_lightcurve_path(self, lightcurve_path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the lightcurve from the path given.

        :param lightcurve_path: The path to the lightcurve file.
        :return: The fluxes and times of the lightcurve
        """
        fluxes, times = self.tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(lightcurve_path)
        nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
        fluxes = np.delete(fluxes, nan_indexes)
        times = np.delete(times, nan_indexes)
        return fluxes, times
