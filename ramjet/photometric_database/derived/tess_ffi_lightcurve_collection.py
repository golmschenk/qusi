"""
Code for a lightcurve collection of the TESS FFI data, as produced by Brian Powell.
"""
import numpy as np
from pathlib import Path
from typing import Iterable

from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


tess_ffi_data_interface = TessFfiDataInterface()


class TessFfiLightcurveCollection(LightcurveCollection):
    """
    A lightcurve collection of the TESS FFI data, as produced by Brian Powell.
    """
    def __init__(self):
        super().__init__()
        self.label = 0

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        return tess_ffi_data_interface.paths_generator_from_sql_table()

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(path)
        return times, fluxes

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        fluxes, times = tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(path)
        magnifications, times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return times, magnifications
