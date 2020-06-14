import numpy as np
from pathlib import Path
from typing import Callable, Iterable, Union, Tuple


class LightcurveCollection:
    """
    A class representing a collection of lightcurves. Used to define how to find, load, and label a set of lightcurves.
    """
    def __init__(self, function_to_get_lightcurve_paths: Callable[[], Iterable[Path]],
                 function_to_load_times_and_fluxes_from_lightcurve_path: Callable[[Path],
                                                                                  Tuple[np.ndarray, np.ndarray]],
                 label: Union[float, None] = None):
        """
        :param function_to_get_lightcurve_paths: A function which returns an iterable of the lightcurve paths.
        :param function_to_load_times_and_fluxes_from_lightcurve_path: A function which, given a lightcurve path, will
                                                                       load the fluxes and times of the lightcurve.
        :param label: The label corresponding to the lightcurves in the collection.
        """
        self.get_lightcurve_paths: Callable[[], Iterable[Path]] = function_to_get_lightcurve_paths
        self.load_times_and_fluxes_from_lightcurve_path: Callable[
            [Path], Tuple[np.ndarray, np.ndarray]] = function_to_load_times_and_fluxes_from_lightcurve_path
        self.label: Union[float, None] = label
