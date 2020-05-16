"""
Code for interfacing with Brian Powell's TESS full frame image (FFI) data.
"""
import pickle
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Union, List


class FfiDataIndexes(Enum):
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


class TessFfiDataInterface:
    """
    A class for interfacing with Brian Powell's TESS full frame image (FFI) data.
    """
    @staticmethod
    def load_fluxes_and_times_from_pickle_file(file_path: Union[Path, str],
                                               flux_type_index: FfiDataIndexes = FfiDataIndexes.CORRECTED_FLUX
                                               ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[flux_type_index.value]
        times = lightcurve[FfiDataIndexes.TIME.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    def get_pickle_directories(self, ffi_root_directory: Path) -> List[Path]:
        """
        Gets the list of pickle containing directories based on the root FFI directory. This function assumes
        Brian Powell's FFI directory structure.

        :param ffi_root_directory: The root FFI directory.
        :return: The list of subdirectories containing the pickle files.
        """
        return list(ffi_root_directory.glob('tesslcs_sector_*/tesslcs_tmag_*_*/'))
