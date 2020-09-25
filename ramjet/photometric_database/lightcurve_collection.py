import numpy as np
from pathlib import Path
from typing import Callable, Iterable, Union, Tuple, List


class LightcurveCollectionMethodNotImplementedError(RuntimeError):
    """
    An error to raise if a collection method that is not implemented is attempted to be used.
    Note, the standard NotImplementedError is not supposed to be used for cases when non-implemented functions are
    meant to be allowed, which is why a custom class is needed.
    """
    pass


class LightcurveCollection:
    """
    A class representing a collection of lightcurves. Used to define how to find, load, and label a set of lightcurves.
    """
    def __init__(self, label: Union[float, None] = None, paths: Union[List[Path], None] = None,
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
        self.paths: Union[List[Path], None] = paths

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        return self.paths

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        raise LightcurveCollectionMethodNotImplementedError

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        raise LightcurveCollectionMethodNotImplementedError

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

    def load_label_from_path(self, path: Path) -> Union[float, np.ndarray]:
        """
        Loads the label of an example from a corresponding path.

        :param path: The path to load the label for.
        :return: The label.
        """
        return self.label

    @staticmethod
    def shuffle_and_split_paths(paths: List[Path], dataset_splits: List[int], number_of_splits: int = 10) -> List[Path]:
        """
        Repeatably shuffles a list of paths and then gets the requested dataset splits from that list of paths.
        Designed to allow splitting a list of paths into training, validation, and testing datasets easily.

        :param paths: The original list of paths.
        :param dataset_splits: The indexes of the dataset splits to return.
        :param number_of_splits: The number of dataset splits.
        :return: The paths of the dataset splits.
        """
        path_array = np.array(paths)
        np.random.seed(0)
        np.random.shuffle(path_array)
        dataset_split_arrays = np.array_split(path_array, number_of_splits)
        dataset_split_arrays_to_keep = [dataset_split_array
                                        for dataset_split_index, dataset_split_array in enumerate(dataset_split_arrays)
                                        if dataset_split_index in dataset_splits]
        paths_array = np.concatenate(dataset_split_arrays_to_keep)
        dataset_split_paths = list(map(Path, paths_array))
        return dataset_split_paths
