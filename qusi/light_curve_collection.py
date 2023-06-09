import random
from pathlib import Path
from typing import Iterable, Callable, Self, Iterator

from qusi.light_curve import LightCurve
from qusi.light_curve_observation import LightCurveObservation


class LightCurveCollection(Iterable):
    """
    :ivar get_paths_function: The function to load the list of paths.
    :ivar load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
    :ivar load_label_from_path_function: The function to load the label for the light curve.
    """

    def __init__(self,
                 get_paths_function: Callable[[], Iterable[Path]],
                 load_times_and_fluxes_from_path_function: Callable[[Path], LightCurve],
                 load_label_from_path_function: Callable[[Path], int]):
        self.get_paths_function: Callable[[], Iterable[Path]] = get_paths_function
        self.load_times_and_fluxes_from_path_function: Callable[[Path], LightCurve] = \
            load_times_and_fluxes_from_path_function
        self.load_label_from_path_function: Callable[[Path], int] = load_label_from_path_function
        self.random = random.Random(0)

    @classmethod
    def new(cls,
            get_paths_function: Callable[[], Iterable[Path]],
            load_times_and_fluxes_from_path_function: Callable[[Path], LightCurve],
            load_label_from_path_function: Callable[[Path], int]
            ) -> Self:
        """
        Creates a new light curve collection.

        :param get_paths_function: The function to load the list of paths.
        :param load_times_and_fluxes_from_path_function: The function to load the times and fluxes from the light curve.
        :param load_label_from_path_function: The function to load the label for the light curve.
        :return: The light curve collection.
        """
        return cls(get_paths_function=get_paths_function,
                   load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function,
                   load_label_from_path_function=load_label_from_path_function)

    @classmethod
    def new_with_label(cls,
                       get_paths_function: Callable[[], Iterable[Path]],
                       load_times_and_fluxes_from_path_function: Callable[[Path], LightCurve],
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
        return cls(get_paths_function=get_paths_function,
                   load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function,
                   load_label_from_path_function=load_label_from_path_function)

    def __iter__(self) -> Iterator[LightCurveObservation]:
        """
        Get the iterable that will iterate through the light curves of the collection.

        :return: The iterable of the light curves.
        """
        light_curve_paths = self.get_shuffled_paths()
        for light_curve_path in light_curve_paths:
            times, fluxes = self.load_times_and_fluxes_from_path_function(light_curve_path)
            label = self.load_label_from_path_function(light_curve_path)
            light_curve = LightCurve.new(times, fluxes)
            light_curve_observation = LightCurveObservation.new(light_curve, label)
            yield light_curve_observation

    def get_shuffled_paths(self) -> Iterable[Path]:
        """
        Gets the shuffled paths iterable.

        :return: The shuffled paths iterable.
        """
        light_curve_paths = list(self.get_paths_function())
        self.random.shuffle(light_curve_paths)
        return light_curve_paths


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
