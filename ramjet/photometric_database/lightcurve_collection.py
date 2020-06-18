import numpy as np
from pathlib import Path
from typing import Callable, Iterable, Union, Tuple


class LightcurveCollection:
    """
    A class representing a collection of lightcurves. Used to define how to find, load, and label a set of lightcurves.
    """
    def __init__(self, label: Union[float, None] = None,
                 function_to_get_paths: Union[Callable[[], Iterable[Path]], None] = None,
                 function_to_load_times_and_fluxes_from_path: Union[
                     Callable[[Path], Tuple[np.ndarray, np.ndarray]], None] = None,
                 function_to_load_times_and_magnifications_from_path: Union[
                     Callable[[Path], Tuple[np.ndarray, np.ndarray]], None] = None):
        """
        :param label: The label corresponding to the lightcurves in the collection.
        :param function_to_get_paths: A function which returns an iterable of the lightcurve paths.
        :param function_to_load_times_and_fluxes_from_path: A function which, given a lightcurve path, will
                                                            load the times and fluxes of the lightcurve.
        :param function_to_load_times_and_magnifications_from_path: A function which, given a lightcurve path, will
                                                                    load the times and magnifications of the lightcurve.
        """
        if function_to_get_paths is not None:
            self.get_paths: Callable[[], Iterable[Path]] = function_to_get_paths
        if function_to_load_times_and_fluxes_from_path is not None:
            self.load_times_and_fluxes_from_path: Callable[
                [Path], Tuple[np.ndarray, np.ndarray]] = function_to_load_times_and_fluxes_from_path
        if function_to_load_times_and_magnifications_from_path is not None:
            self.load_times_and_magnifications_from_path: Callable[
                [Path], Tuple[np.ndarray, np.ndarray]] = function_to_load_times_and_magnifications_from_path
        self.label: Union[float, None] = label

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        raise NotImplementedError

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        raise NotImplementedError

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        raise NotImplementedError

    @staticmethod
    def generate_synthetic_signal_from_real_data(fluxes: np.ndarray, times: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Takes real lightcurve data and converts it to a form that can be used for synthetic lightcurve injection.

        :param fluxes: The real lightcurve fluxes.
        :param times: The real lightcurve times.
        :return: Fake synthetic magnifications and times.
        """
        fluxes -= np.minimum(np.min(fluxes), 0)  # Fix negative flux cases if they exist.
        flux_median = np.median(fluxes)
        normalized_fluxes = fluxes / flux_median
        relative_times = times - np.min(times)
        return normalized_fluxes, relative_times
