from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ramjet.photometric_database.light_curve_collection import LightCurveCollection


class SimpleLightCurveCollection(LightCurveCollection):
    """
    A simple positive and negative directory based light curve collection.
    """

    def __init__(self, collection_directory: Path = Path("data/simple_test_dataset")):
        super().__init__()
        self.collection_directory = collection_directory

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        return self.collection_directory.glob("**/*.npz")

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        contents = np.load(str(path))
        times = contents["times"]
        fluxes = contents["fluxes"]
        return times, fluxes
