import re

from peewee import Select

from ramjet.data_interface.tess_ffi_light_curve_metadata_manager import TessFfiLightCurveMetadata
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum
from pathlib import Path
from typing import Iterable, Union, List

import numpy as np
import pandas as pd

from ramjet.photometric_database.light_curve_collection import LightCurveCollection


class ColumnName(StrEnum):
    TIME__DAYS = 'time__days'
    MAGNIFICATION = 'magnification'


class SiddhantSolankiHeartBeatSyntheticSignalsCollection(LightCurveCollection):
    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/siddhant_solanki_synthetic_signals')
        self.label = 1

    def get_paths(self) -> Iterable[Path]:
        all_synthetic_signal_paths = self.data_directory.glob('*.txt')
        heart_beat_synthetic_signals = [path for path in all_synthetic_signal_paths
                                            if re.match(r'generated_lc_\d+.txt', path.name) is not None]
        return heart_beat_synthetic_signals

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        synthetic_signal_data_frame = pd.read_csv(path, names=[ColumnName.MAGNIFICATION],
                                                  skipinitialspace=True, delim_whitespace=True, skiprows=1)
        synthetic_signal_data_frame.dropna(inplace=True)
        magnifications = synthetic_signal_data_frame[ColumnName.MAGNIFICATION].values
        step_size__days = 0.0069444444
        times = np.arange(0, magnifications.shape[0] * step_size__days, step_size__days)
        assert times.shape[0] == magnifications.shape[0]
        return times, magnifications


class SiddhantSolankiNonHeartBeatSyntheticSignalsCollection(LightCurveCollection):
    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/siddhant_solanki_synthetic_signals')
        self.label = 0

    def get_paths(self) -> Iterable[Path]:
        all_synthetic_signal_paths = self.data_directory.glob('*.txt')
        non_heart_beat_synthetic_signals = [path for path in all_synthetic_signal_paths
                                            if re.match(r'generated_lc_fake_\d+.txt', path.name) is not None]
        return non_heart_beat_synthetic_signals

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        synthetic_signal_data_frame = pd.read_csv(path, names=[ColumnName.MAGNIFICATION],
                                                  skipinitialspace=True, delim_whitespace=True, skiprows=1)
        synthetic_signal_data_frame.dropna(inplace=True)
        magnifications = synthetic_signal_data_frame[ColumnName.MAGNIFICATION].values
        step_size__days = 0.0069444444
        times = np.arange(0, magnifications.shape[0] * step_size__days, step_size__days)
        assert times.shape[0] == magnifications.shape[0]
        return times, magnifications

class TessFfiHeartBeatHardNegativeLightcurveCollection(TessFfiLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing eclipsing binaries.
    """
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 0
        self.hard_negative_ids = list(pd.read_csv('data/heart_beat_hard_negatives.csv')['tic_id'].values)

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.
        :return: The SQL query.
        """
        query = super().get_sql_query()
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(self.hard_negative_ids))
        return query
