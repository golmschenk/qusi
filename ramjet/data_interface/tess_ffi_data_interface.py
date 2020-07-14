"""
Code for interfacing with Brian Powell's TESS full frame image (FFI) data.
"""
import re
import pickle
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Union


class FfiDataIndexes(Enum):
    """
    An enum for accessing Brian Powell's FFI pickle data with understandable indexes.
    """
    TIC_ID = 0
    RA = 1
    DEC = 2
    TESS_MAGNITUDE = 3
    CAMERA = 4
    CHIP = 5
    TIME__BTJD = 6
    RAW_FLUX = 7
    CORRECTED_FLUX = 8
    PCA_FLUX = 9
    FLUX_ERROR = 10
    QUALITY = 11


class TessFfiDataInterface:
    """
    A class for interfacing with Brian Powell's TESS full frame image (FFI) data.
    """

    def __init__(self, lightcurve_root_directory_path: Path = Path('data/tess_ffi_lightcurves'),
                 database_path: Union[Path, str] = Path('data/metadatabase.sqlite3')):
        self.lightcurve_root_directory_path: Path = lightcurve_root_directory_path
        self.database_path: Union[Path, str] = database_path

    @staticmethod
    def load_fluxes_and_times_from_pickle_file(file_path: Union[Path, str],
                                               flux_type_index: FfiDataIndexes = FfiDataIndexes.CORRECTED_FLUX
                                               ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_type_index: The flux type to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[flux_type_index.value]
        times = lightcurve[FfiDataIndexes.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    @staticmethod
    def load_fluxes_flux_errors_and_times_from_pickle_file(
            file_path: Union[Path, str], flux_type_index: FfiDataIndexes = FfiDataIndexes.CORRECTED_FLUX
    ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes, flux errors, and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_type_index: The flux type to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[flux_type_index.value]
        flux_errors = lightcurve[FfiDataIndexes.FLUX_ERROR.value]
        times = lightcurve[FfiDataIndexes.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        assert times.shape == flux_errors.shape
        return fluxes, flux_errors, times

    @staticmethod
    def get_tic_id_and_sector_from_file_path(file_path: Union[Path, str]) -> (int, int):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param file_path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_(\d+)(?:_104)?/tesslcs_tmag_\d+_\d+/tesslc_(\d+)', file_path)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Search for Brian Powell's FFI path convention with only the file name containing the target.
        # E.g., "tesslc_290374453"
        match = re.search(r'tesslc_(\d+)', file_path)
        if match:
            return int(match.group(1)), None
        # Raise an error if none of the patterns matched.
        raise ValueError(f'{file_path} does not match a known pattern to extract TIC ID and sector from.')

    @staticmethod
    def get_floor_magnitude_from_file_path(file_path: Union[Path, str]) -> int:
        """
        Gets the floor magnitude from the FFI file path.

        :param file_path: The path of the file to extract the magnitude.
        :return: The magnitude floored.
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_\d+(?:_104)?/tesslcs_tmag_(\d+)_\d+/tesslc_\d+', file_path)
        if match:
            return int(match.group(1))
        raise ValueError(f'{file_path} does not match a known pattern to extract magnitude from.')

    @staticmethod
    def get_magnitude_from_file(file_path: Union[Path, str]) -> float:
        """
        Loads the magnitude from the file.

        :param file_path: The path to the file.
        :return: The magnitude of the target.
        """
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        magnitude = lightcurve[FfiDataIndexes.TESS_MAGNITUDE.value]
        return magnitude
