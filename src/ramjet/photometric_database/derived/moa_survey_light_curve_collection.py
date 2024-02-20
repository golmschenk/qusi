from __future__ import annotations

import re
import shutil
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import scipy.stats
from filelock import FileLock

from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.light_curve_collection import LightCurveCollection

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np


class MoaSurveyLightCurveCollection(LightCurveCollection):
    """
    A collection of light curves based on the MOA 9-year survey.
    """

    moa_data_interface = MoaDataInterface()

    def __init__(
        self,
        survey_tags: list[str],
        dataset_splits: list[int] | None = None,
        label: float | list[float] | np.ndarray | None = None,
    ):
        super().__init__()
        self.label = label
        self.survey_tags: list[str] = survey_tags
        self.dataset_splits: list[int] | None = dataset_splits

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        paths: list[Path] = []
        for tag in self.survey_tags:
            tag_paths = self.moa_data_interface.survey_tag_to_path_list_dictionary[tag]
            if self.dataset_splits is not None:
                # Split on each tag, so that the splitting remains across collections with different tag selections.
                tag_paths = self.shuffle_and_split_paths(tag_paths, self.dataset_splits)
            paths.extend(tag_paths)
        return paths

    def move_path_to_nvme(self, path: Path) -> Path:
        match = re.match(r"gpu\d{3}", socket.gethostname())
        if match is not None:
            nvme_path = Path("/lscratch/golmsche").joinpath(path)
            if not nvme_path.exists():
                nvme_path.parent.mkdir(exist_ok=True, parents=True)
                nvme_lock_path = nvme_path.parent.joinpath(nvme_path.name + ".lock")
                lock = FileLock(str(nvme_lock_path))
                with lock.acquire():
                    if not nvme_path.exists():
                        nvme_tmp_path = nvme_path.parent.joinpath(nvme_path.name + ".tmp")
                        shutil.copy(path, nvme_tmp_path)
                        nvme_tmp_path.rename(nvme_path)
            return nvme_path
        return path

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        path = self.move_path_to_nvme(path)
        light_curve_dataframe = pd.read_feather(path)
        times = light_curve_dataframe["HJD"].values
        fluxes = light_curve_dataframe["flux"].values
        return times, fluxes

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the light curve/signal file.
        :return: The times and the magnifications of the light curve/signal.
        """
        path = self.move_path_to_nvme(path)
        times, fluxes = self.load_times_and_fluxes_from_path(path)
        magnifications, times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return times, magnifications

    @staticmethod
    def generate_synthetic_signal_from_real_data(fluxes: np.ndarray, times: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Takes real light curve data and converts it to a form that can be used for synthetic light curve injection.

        :param fluxes: The real light curve fluxes.
        :param times: The real light curve times.
        :return: Fake synthetic magnifications and times.
        """
        flux_median_absolute_deviation = scipy.stats.median_abs_deviation(fluxes)
        normalized_fluxes = (fluxes / flux_median_absolute_deviation) * 0.25
        # relative_times = times - np.min(times)
        return normalized_fluxes, times
