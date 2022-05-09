import re

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum
from pathlib import Path
from typing import Iterable

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
        synthetic_signal_data_frame = pd.read_csv(path, names=[ColumnName.TIME__DAYS, ColumnName.MAGNIFICATION],
                                                  skipinitialspace=True, delim_whitespace=True, skiprows=1)
        synthetic_signal_data_frame.dropna(inplace=True)
        times = synthetic_signal_data_frame[ColumnName.TIME__DAYS].values
        magnifications = synthetic_signal_data_frame[ColumnName.MAGNIFICATION].values
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
        synthetic_signal_data_frame = pd.read_csv(path, names=[ColumnName.TIME__DAYS, ColumnName.MAGNIFICATION],
                                                  skipinitialspace=True, delim_whitespace=True)
        synthetic_signal_data_frame.dropna(inplace=True)
        times = synthetic_signal_data_frame[ColumnName.TIME__DAYS].values
        magnifications = synthetic_signal_data_frame[ColumnName.MAGNIFICATION].values
        return times, magnifications