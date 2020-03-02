"""
Code to represent a database to train to find exoplanet transits in FFI data based on known TOI dispositions.
"""
import pickle
from typing import Union

import numpy as np
from enum import Enum
from pathlib import Path

from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase


class FfiDataIndixes(Enum):
    """
    An enum for accessing Brian Powell's FFI pickle data with understandable indexes.
    """
    TIC_ID = 0
    RA = 1
    DEC = 2
    TESS_MAGNITUDE = 3
    TIME = 4
    RAW_FLUX = 5
    CORRECTED_FLUX = 6
    PCA_FLUX = 7
    FLUX_ERROR = 8


class FfiToiDatabase(TessSyntheticInjectedDatabase):
    """
    Code to represent a database to train to find exoplanet transits in FFI data based on known TOI dispositions.
    """
    @staticmethod
    def load_fluxes_and_times_from_ffi_pickle_file(file_path: Union[Path, str]) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[FfiDataIndixes.CORRECTED_FLUX.value]
        times = lightcurve[FfiDataIndixes.TIME.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    @staticmethod
    def generate_synthetic_signal_from_real_data(fluxes: np.ndarray, times: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Takes real lightcurve data and converts it to a form that can be used for synthetic lightcurve injection.

        :param fluxes: The real lightcurve fluxes.
        :param times: The real lightcurve times.
        :return: Fake synthetic magnifications and times.
        """
        flux_median = np.median(fluxes)
        normalized_fluxes = fluxes / flux_median
        relative_times = times - np.min(times)
        return normalized_fluxes, relative_times
