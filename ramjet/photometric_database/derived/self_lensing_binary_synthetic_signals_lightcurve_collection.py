"""
Code for a lightcurve collection of Agnieszka Cieplak's synthetic signals.
"""
import re
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


class SelfLensingBinarySyntheticSignalsLightcurveCollection(LightcurveCollection):
    """
    A lightcurve collection for Agnieszka Cieplak's synthetic signals.
    """

    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/self_lensing_binary_synthetic_signals')
        self.label = 1

    def download_csv_files(self):
        """
        Downloads Agnieszka Cieplak's synthetic signals in their original CSV form.
        """
        print('Downloading synthetic signal CSV files...')
        tar_file_path = self.data_directory.joinpath('synthetic_signals_csv_files.tar')
        urllib.request.urlretrieve('https://api.onedrive.com/v1.0/shares/s!AjiSFm1N8Bv7ghXushB7JOzABXdv/root/content',
                                   str(tar_file_path))
        with tarfile.open(tar_file_path) as csv_tar_file:
            csv_tar_file.extractall(self.data_directory)
        tar_file_path.unlink()
        csv_uncompressed_directory = self.data_directory.joinpath('LearningSetedgeon_all_sum')
        for path in csv_uncompressed_directory.glob('*'):
            path.rename(self.data_directory.joinpath(path.name))
        csv_uncompressed_directory.rmdir()

    def convert_csv_files_to_project_format(self):
        """
        Converts Agnieszka Cieplak's synthetic signal CSV files to the project format feather files.
        """
        print('Converting synthetic signals to project format...')
        out_paths = self.data_directory.glob('*.out')
        synthetic_signal_csv_paths = [path for path in out_paths if re.match(r'lc_\d+\.out', path.name)]
        for synthetic_signal_csv_path in synthetic_signal_csv_paths:
            synthetic_signal = pd.read_csv(synthetic_signal_csv_path, names=['time__days', 'magnification'],
                                           delim_whitespace=True, skipinitialspace=True)
            synthetic_signal.to_feather(self.data_directory.joinpath(f'{synthetic_signal_csv_path.stem}.feather'))
            synthetic_signal_csv_path.unlink()

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        return self.data_directory.glob('*.feather')

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        synthetic_signal_data_frame = pd.read_feather(path)
        times = synthetic_signal_data_frame['time__days'].values
        magnifications = synthetic_signal_data_frame['magnification'].values
        return times, magnifications


class ReversedSelfLensingBinarySyntheticSignalsLightcurveCollection(
    SelfLensingBinarySyntheticSignalsLightcurveCollection):
    """
    A lightcurve collection for a time reversed version Agnieszka Cieplak's synthetic signals. Useful as a negative
    training dataset, as the lensing signals will only occur in a specific order relative to the other signal
    components.
    """

    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/self_lensing_binary_synthetic_signals')
        self.label = 0

    @staticmethod
    def reverse_signal(times: np.ndarray, magnitudes: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Reverses the signal (i.e., the original first magnitude is in the last position and vice versa).

        :param times: The times of the original signal.
        :param magnitudes: The magnitudes of the original signal.
        :return: The reversed signal times and magnitudes.
        """
        return times, magnitudes[::-1]

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        original_times, original_magnitudes = super().load_times_and_magnifications_from_path(path)
        times, magnifications = self.reverse_signal(original_times, original_magnitudes)
        return times, magnifications


if __name__ == '__main__':
    lightcurve_collection = SelfLensingBinarySyntheticSignalsLightcurveCollection()
    lightcurve_collection.data_directory.mkdir(parents=True, exist_ok=True)
    lightcurve_collection.download_csv_files()
    lightcurve_collection.convert_csv_files_to_project_format()
    print('Self lensing binary synthetic signal lightcurve collection ready.')
