"""
Code to for a class to represent a TESS FFI light curve.
"""
from __future__ import annotations

import pickle
import re
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Union, List

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.photometric_database.tess_light_curve import TessLightCurve


class TessFfiColumnName(Enum):
    """
    An enum to represent the column names of the TESS FFI data.
    """
    TIME__BTJD = 'time__btjd'
    CORRECTED_FLUX = 'corrected_flux'
    RAW_FLUX = 'raw_flux'
    FLUX_ERROR = 'flux_error'


class TessFfiPickleIndex(Enum):
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


class TessFfiLightCurve(TessLightCurve):
    """
    A class to for a class to represent a TESS FFI light curve.
    """
    mast_tess_data_interface = TessDataInterface()

    def __init__(self):
        super().__init__()
        self.flux_column_names = [TessFfiColumnName.CORRECTED_FLUX.value,
                                  TessFfiColumnName.RAW_FLUX.value]

    @classmethod
    def from_path(cls, path: Path, column_names_to_load: Union[List[TessFfiColumnName], None] = None
                  ) -> TessFfiLightCurve:
        """
        Creates an FFI TESS light curve from a path to one of Brian Powell's pickle files.

        :param path: The path to the pickle file to load.
        :param column_names_to_load: The FFI light curve columns to load from the pickle file. By default, all will be
                                     loaded. Selecting specific ones may speed the process when loading many light
                                     curves.
        :return: The light curve.
        """
        light_curve = cls()
        light_curve.time_column_name = TessFfiColumnName.TIME__BTJD.value
        if column_names_to_load is None:
            column_names_to_load = list(TessFfiColumnName)
        with path.open('rb') as pickle_file:
            light_curve_data_dictionary = pickle.load(pickle_file)
            for column_name in column_names_to_load:
                pickle_index = TessFfiPickleIndex[column_name.name]
                light_curve.data_frame[column_name.value] = light_curve_data_dictionary[pickle_index.value]
        light_curve.tic_id, light_curve.sector = light_curve.get_tic_id_and_sector_from_file_path(path)
        return light_curve

    @staticmethod
    def get_tic_id_and_sector_from_file_path(path: Union[Path, str]) -> (int, Union[int, None]):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        if isinstance(path, Path):
            path = str(path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_(\d+)(?:_104)?/tesslcs_tmag_\d+_\d+/tesslc_(\d+)', path)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Search for Brian Powell's FFI path convention with only the file name containing the target.
        # E.g., "tesslc_290374453"
        match = re.search(r'tesslc_(\d+)', path)
        if match:
            return int(match.group(1)), None
        # Raise an error if none of the patterns matched.
        raise ValueError(f'{path} does not match a known pattern to extract TIC ID and sector from.')


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
            light_curve = pickle.load(pickle_file)
        magnitude = light_curve[TessFfiPickleIndex.TESS_MAGNITUDE.value]
        return magnitude

    @classmethod
    def load_fluxes_and_times_from_pickle_file(
            cls, file_path: Union[Path, str], flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX
        ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_column_name: The flux type to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        light_curve = cls.from_path(file_path, column_names_to_load=[TessFfiColumnName.TIME__BTJD,
                                                                     flux_column_name])
        fluxes = light_curve.data_frame[flux_column_name.value]
        times = light_curve.data_frame[TessFfiColumnName.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    @classmethod
    def load_fluxes_flux_errors_and_times_from_pickle_file(
            cls, file_path: Union[Path, str], flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Loads the fluxes, flux errors, and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_column_name: The flux type to load.
        :return: The fluxes, flux errors, and times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        light_curve = cls.from_path(file_path, column_names_to_load=[TessFfiColumnName.TIME__BTJD,
                                                                     TessFfiColumnName.FLUX_ERROR,
                                                                     flux_column_name])
        fluxes = light_curve.data_frame[flux_column_name.value]
        flux_errors = light_curve.data_frame[TessFfiColumnName.FLUX_ERROR.value]
        times = light_curve.data_frame[TessFfiColumnName.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        return fluxes, flux_errors, times
