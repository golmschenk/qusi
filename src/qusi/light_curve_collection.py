from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from qusi.light_curve import LightCurve
from qusi.light_curve_observation import LightCurveObservation

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class LightCurveCollectionBase(ABC):
    @abstractmethod
    def light_curve_iter(self) -> Iterator[LightCurve]:
        pass

    @abstractmethod
    def load_times_and_fluxes_from_path(self, path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        pass


class LightCurveObservationCollectionBase(LightCurveCollectionBase):
    @abstractmethod
    def observation_iter(self) -> Iterator[LightCurveObservation]:
        pass

class LightCurveObservationIndexableBase(ABC):
    @abstractmethod
    def __getitem__(self, indexes: int | tuple[int]) -> LightCurveObservation | tuple[LightCurveObservation]:
        pass

class PathIterableBase(ABC):
    @abstractmethod
    def get_shuffled_paths(self) -> list[Path]:
        pass

    @abstractmethod
    def get_paths(self) -> list[Path]:
        pass


class PathIndexableBase(ABC):
    @abstractmethod
    def __getitem__(self, indexes: int | tuple[int]) -> Path | tuple[Path]:
        pass


class PathGetterBase(PathIterableBase, PathIndexableBase):
    pass


@dataclass
class PathGetter(PathGetterBase):
    """
    A class to get paths from a path generation function.

    :ivar get_paths_function: The function which returns the path iterable.
    :ivar random_number_generator: A random number generator.
    """
    get_paths_function: Callable[[], Iterable[Path]]
    random_number_generator: Random

    @classmethod
    def new(cls, get_paths_function: Callable[[], Iterable[Path]]) -> Self:
        random_number_generator = Random(0)
        instance = cls(get_paths_function=get_paths_function, random_number_generator=random_number_generator)
        return instance

    def get_shuffled_paths(self) -> Iterable[Path]:
        """
        Gets the shuffled paths iterable.

        :return: The shuffled paths iterable.
        """
        light_curve_paths = self.get_paths()
        self.random_number_generator.shuffle(light_curve_paths)
        return light_curve_paths

    def get_paths(self):
        """
        Gets the paths iterable.

        :return: The paths iterable.
        """
        light_curve_paths = list(self.get_paths_function())
        return light_curve_paths

    def __getitem__(self, index: int | tuple[int]) -> Path | tuple[Path]:
        light_curve_paths = self.get_paths()
        indexed_light_curve_paths = np.array(light_curve_paths)[index]
        if isinstance(indexed_light_curve_paths, Path):
            return indexed_light_curve_paths
        else:
            return indexed_light_curve_paths.tolist()


@dataclass
class LightCurveCollection(LightCurveCollectionBase, LightCurveObservationIndexableBase):
    """
    :ivar path_getter: The PathIterableBase object for the collection.
    :ivar load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
    """
    path_getter: PathGetterBase
    load_times_and_fluxes_from_path_function: Callable[
        [Path], tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]

    @classmethod
    def new(cls,
            get_paths_function: Callable[[], Iterable[Path]],
            load_times_and_fluxes_from_path_function: Callable[
                [Path], tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
            ) -> Self:
        """
        Creates a new light curve collection.

        :param get_paths_function: The function to load the list of paths.
        :param load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
        :return: The light curve collection.
        """
        path_getter = PathGetter.new(get_paths_function=get_paths_function)
        return cls(path_getter=path_getter,
                   load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function)

    def light_curve_iter(self) -> Iterator[LightCurve]:
        """
        Get the iterable that will iterate through the light curves of the collection.

        :return: The iterable of the light curves.
        """
        light_curve_paths = self.path_getter.get_shuffled_paths()
        for light_curve_path in light_curve_paths:
            times, fluxes = self.load_times_and_fluxes_from_path_function(light_curve_path)
            light_curve = LightCurve.new(times, fluxes)
            yield light_curve

    def load_times_and_fluxes_from_path(self, path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self.load_times_and_fluxes_from_path_function(path)

    def __getitem__(self, index: int) -> LightCurve:
        light_curve_path = self.path_getter[index]
        times, fluxes = self.load_times_and_fluxes_from_path(light_curve_path)
        light_curve = LightCurve.new(times, fluxes)
        return light_curve


@dataclass
class LabeledLightCurveCollection(LightCurveObservationCollectionBase, LightCurveObservationIndexableBase):
    """
    :ivar path_getter: The PathGetterBase object for the collection.
    :ivar light_curve_collection: The LightCurveCollectionBase object for the collection.
    :ivar load_label_from_path_function: The function to load the label for the light curve.
    """
    path_getter: PathGetterBase
    light_curve_collection: LightCurveCollectionBase
    load_label_from_path_function: Callable[[Path], int]

    @classmethod
    def new(cls,
            get_paths_function: Callable[[], Iterable[Path]],
            load_times_and_fluxes_from_path_function: Callable[
                [Path], tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
            load_label_from_path_function: Callable[[Path], int]
            ) -> Self:
        """
        Creates a new light curve collection.

        :param get_paths_function: The function to load the list of paths.
        :param load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
        :param load_label_from_path_function: The function to load the label for the light curve.
        :return: The light curve collection.
        """
        path_iterable = PathGetter.new(get_paths_function=get_paths_function)
        light_curve_collection = LightCurveCollection(
            path_getter=path_iterable,
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function)
        return cls(path_getter=path_iterable,
                   light_curve_collection=light_curve_collection,
                   load_label_from_path_function=load_label_from_path_function)

    @classmethod
    def new_with_label(cls,
                       get_paths_function: Callable[[], Iterable[Path]],
                       load_times_and_fluxes_from_path_function: Callable[
                           [Path], tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
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
        path_iterable = PathGetter.new(get_paths_function=get_paths_function)
        light_curve_collection = LightCurveCollection(
            path_getter=path_iterable,
            load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function)
        return cls(path_getter=path_iterable,
                   light_curve_collection=light_curve_collection,
                   load_label_from_path_function=load_label_from_path_function)

    def light_curve_iter(self) -> Iterator[LightCurve]:
        return self.light_curve_collection.light_curve_iter()

    def load_times_and_fluxes_from_path(self, path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self.light_curve_collection.load_times_and_fluxes_from_path(path=path)

    def observation_iter(self) -> Iterator[LightCurveObservation]:
        """
        Get the iterable that will iterate through the light curves of the collection.

        :return: The iterable of the light curves.
        """
        light_curve_paths = self.path_getter.get_shuffled_paths()
        for light_curve_path in light_curve_paths:
            times, fluxes = self.light_curve_collection.load_times_and_fluxes_from_path(light_curve_path)
            label = self.load_label_from_path_function(light_curve_path)
            light_curve = LightCurve.new(times, fluxes)
            light_curve_observation = LightCurveObservation.new(light_curve, label)
            yield light_curve_observation

    def __getitem__(self, index: int) -> LightCurveObservation:
        light_curve_path = self.path_getter[index]
        times, fluxes = self.light_curve_collection.load_times_and_fluxes_from_path(light_curve_path)
        label = self.load_label_from_path_function(light_curve_path)
        light_curve = LightCurve.new(times, fluxes)
        light_curve_observation = LightCurveObservation.new(light_curve, label)
        return light_curve_observation


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
