from random import Random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Callable, Self, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt

from qusi.light_curve import LightCurve
from qusi.light_curve_observation import LightCurveObservation


class LightCurveCollectionBase(ABC):
    @abstractmethod
    def light_curve_iter(self) -> Iterator[LightCurve]:
        pass

    @abstractmethod
    def load_times_and_fluxes_from_path(self, path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        pass


class LightCurveObservationCollectionBase(LightCurveCollectionBase):
    @abstractmethod
    def observation_iter(self) -> Iterator[LightCurveObservation]:
        pass


class PathIterableBase(ABC):
    @abstractmethod
    def get_shuffled_paths(self) -> List[Path]:
        pass


class PathIterable(PathIterableBase):
    def __init__(self,
                 get_paths_function: Callable[[], Iterable[Path]],
                 random_number_generator: Random):
        self.get_paths_function: Callable[[], Iterable[Path]] = get_paths_function
        self.random_number_generator = random_number_generator

    def get_shuffled_paths(self) -> Iterable[Path]:
        """
        Gets the shuffled paths iterable.

        :return: The shuffled paths iterable.
        """
        light_curve_paths = list(self.get_paths_function())
        self.random_number_generator.shuffle(light_curve_paths)
        return light_curve_paths


class LightCurveCollection(LightCurveCollectionBase):
    """
    :ivar path_iterable: The PathIterableBase object for the collection.
    :ivar load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
    """

    def __init__(self,
                 path_iterable: PathIterableBase,
                 load_times_and_fluxes_from_path_function: Callable[
                     [Path], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]):
        self.path_iterable: PathIterableBase = path_iterable
        self.load_times_and_fluxes_from_path_function: Callable[
            [Path], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = \
            load_times_and_fluxes_from_path_function

    @classmethod
    def new(cls,
            get_paths_function: Callable[[], Iterable[Path]],
            load_times_and_fluxes_from_path_function: Callable[
                [Path], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
            ) -> Self:
        """
        Creates a new light curve collection.

        :param get_paths_function: The function to load the list of paths.
        :param load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
        :return: The light curve collection.
        """
        random_number_generator = Random(0)
        path_iterable = PathIterable(get_paths_function=get_paths_function,
                                     random_number_generator=random_number_generator)
        return cls(path_iterable=path_iterable,
                   load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function)

    def light_curve_iter(self) -> Iterator[LightCurve]:
        """
        Get the iterable that will iterate through the light curves of the collection.

        :return: The iterable of the light curves.
        """
        light_curve_paths = self.path_iterable.get_shuffled_paths()
        for light_curve_path in light_curve_paths:
            times, fluxes = self.load_times_and_fluxes_from_path_function(light_curve_path)
            light_curve = LightCurve.new(times, fluxes)
            yield light_curve

    def load_times_and_fluxes_from_path(self, path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self.load_times_and_fluxes_from_path_function(path)


class LabeledLightCurveCollection(LightCurveObservationCollectionBase):
    """
    :ivar path_iterable: The PathIterableBase object for the collection.
    :ivar light_curve_collection: The LightCurveCollectionBase object for the collection.
    :ivar load_label_from_path_function: The function to load the label for the light curve.
    """

    def __init__(self,
                 path_iterable: PathIterableBase,
                 light_curve_collection: LightCurveCollectionBase,
                 load_label_from_path_function: Callable[[Path], int]):
        self.path_iterable: PathIterableBase = path_iterable
        self.light_curve_collection: LightCurveCollectionBase = light_curve_collection
        self.load_label_from_path_function: Callable[[Path], int] = load_label_from_path_function

    @classmethod
    def new(cls,
            get_paths_function: Callable[[], Iterable[Path]],
            load_times_and_fluxes_from_path_function: Callable[
                [Path], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
            load_label_from_path_function: Callable[[Path], int]
            ) -> Self:
        """
        Creates a new light curve collection.

        :param get_paths_function: The function to load the list of paths.
        :param load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
        :param load_label_from_path_function: The function to load the label for the light curve.
        :return: The light curve collection.
        """
        random_number_generator = Random(0)
        path_iterable = PathIterable(get_paths_function=get_paths_function,
                                     random_number_generator=random_number_generator)
        light_curve_collection = LightCurveCollection(
            path_iterable=path_iterable,
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function)
        return cls(path_iterable=path_iterable,
                   light_curve_collection=light_curve_collection,
                   load_label_from_path_function=load_label_from_path_function)

    @classmethod
    def new_with_label(cls,
                       get_paths_function: Callable[[], Iterable[Path]],
                       load_times_and_fluxes_from_path_function: Callable[
                           [Path], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
                       label: int
                       ) -> Self:
        """
        Creates a new light curve collection with a specific label for all light curves..

        :param get_paths_function: The function to load the list of paths.
        :param load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
        :param label: The label to be applied to all light curves in the collection.
        :return: The light curve collection.
        """
        load_label_from_path_function = create_constant_label_for_path_function(label)
        random_number_generator = Random(0)
        path_iterable = PathIterable(get_paths_function=get_paths_function,
                                     random_number_generator=random_number_generator)
        light_curve_collection = LightCurveCollection(
            path_iterable=path_iterable,
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function)
        return cls(path_iterable=path_iterable,
                   light_curve_collection=light_curve_collection,
                   load_label_from_path_function=load_label_from_path_function)

    def light_curve_iter(self) -> Iterator[LightCurve]:
        return self.light_curve_collection.light_curve_iter()

    def load_times_and_fluxes_from_path(self, path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self.light_curve_collection.load_times_and_fluxes_from_path(path=path)

    def observation_iter(self) -> Iterator[LightCurveObservation]:
        """
        Get the iterable that will iterate through the light curves of the collection.

        :return: The iterable of the light curves.
        """
        light_curve_paths = self.path_iterable.get_shuffled_paths()
        for light_curve_path in light_curve_paths:
            times, fluxes = self.light_curve_collection.load_times_and_fluxes_from_path(light_curve_path)
            label = self.load_label_from_path_function(light_curve_path)
            light_curve = LightCurve.new(times, fluxes)
            light_curve_observation = LightCurveObservation.new(light_curve, label)
            yield light_curve_observation


def create_constant_label_for_path_function(label: int) -> Callable[[Path], int]:
    """
    Creates a closure function that accepts a path, but always returns a given label regardless of the path.

    :param label: The label that the closure function should return.
    :return: The closure function.
    """

    def constant_label_for_path(_path: Path) -> int:
        """
        The function which will return the outer label regardless of path.

        :param _path: The unused path.
        :return: The label
        """
        return label

    return constant_label_for_path
