"""
Code for interfacing with Brian Powell's TESS full frame image (FFI) data.
"""
import pickle
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Union, List, Iterable


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

    @staticmethod
    def get_pickle_directories(ffi_root_directory: Path) -> List[Path]:
        """
        Gets the list of pickle containing directories based on the root FFI directory. This function assumes
        Brian Powell's FFI directory structure.

        :param ffi_root_directory: The root FFI directory.
        :return: The list of subdirectories containing the pickle files.
        """
        return list(ffi_root_directory.glob('tesslcs_sector_*/tesslcs_tmag_*_*/'))

    @staticmethod
    def create_path_list_pickle_repeating_generator(paths: List[Path]) -> Iterable[Path]:
        """
        Creates a generator for a list of paths, where each path has it's pickle files repeatedly iterated over.

        :param paths: The list of paths containing pickle files.
        :return: The resulting generator.
        """
        generator_dictionary = {}
        for path in paths:  # Create a generator for each, so long as there's at least 1 pickle file.
            try:
                next(path.glob('.pkl'))  # If this doesn't fail, there's at least 1 pickle file in the directory.
                generator_dictionary[path] = path.glob('.pkl')
            except StopIteration:
                continue

        def glob_dictionary_generator():
            """The generator to return."""
            current_loop_generator_dictionary = generator_dictionary
            while True:
                next_loop_generator_dictionary = {}
                for path_, glob_generator in current_loop_generator_dictionary.items():
                    try:
                        yield next(glob_generator)
                    except StopIteration:  # Repeat the generator if it ran out.
                        glob_generator = path_.glob('.pkl')
                        yield next(glob_generator)
                    next_loop_generator_dictionary[path_] = glob_generator
                current_loop_generator_dictionary = next_loop_generator_dictionary

        return glob_dictionary_generator()

    def create_subdirectories_pickle_repeating_generator(self, ffi_root_directory: Path) -> Iterable[Path]:
        """
        Creates a generator for the pickle subdirectories, where each path has it's pickle files repeatedly iterated
        over. Each directory is sampled from equally. This function assumes Brian Powell's FFI directory structure.

        :param ffi_root_directory: The root FFI directory.
        :return: The resulting generator.
        """
        pickle_subdirectories = self.get_pickle_directories(ffi_root_directory)
        generator = self.create_path_list_pickle_repeating_generator(pickle_subdirectories)
        return generator
