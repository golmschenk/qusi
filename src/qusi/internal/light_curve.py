from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from public import public
from typing_extensions import Self


@public
@dataclass
class LightCurve:
    """
    A class to represent a light curve. A light curve is a collection of data which may includes times, fluxes,
    and related values.

    :ivar times: The times of the light curve.
    :ivar fluxes: The fluxes of the light curve.
    """

    times: npt.NDArray[np.float32]
    fluxes: npt.NDArray[np.float32]

    @classmethod
    def new(cls, times: npt.NDArray[np.float32], fluxes: npt.NDArray[np.float32]) -> Self:
        """
        Creates a new light curve.

        :param times: The times of the light curve.
        :param fluxes: The fluxes of the light curve.
        :return: The light curve.
        """
        return cls(times=times, fluxes=fluxes)


def remove_nan_flux_data_points_from_light_curve(light_curve: LightCurve) -> LightCurve:
    """
    Removes the NaN values from a light curve in a light curve. If there is a NaN in either the times or the
    fluxes, both corresponding values are removed.

    :param light_curve: The light curve.
    :return: The light curve with NaN values removed.
    """
    light_curve = deepcopy(light_curve)
    nan_flux_indexes = np.isnan(light_curve.fluxes)
    light_curve.fluxes = light_curve.fluxes[~nan_flux_indexes]
    light_curve.times = light_curve.times[~nan_flux_indexes]
    return light_curve


def randomly_roll_light_curve(light_curve: LightCurve) -> LightCurve:
    """
    Randomly rolls a light curve. That is, a random position in the light curve is chosen, the light curve
    is split at that point, and the order of the two halves are swapped.

    :param light_curve: The light curve.
    :return: The rolled light curve.
    """
    light_curve = deepcopy(light_curve)
    random_index = np.random.randint(light_curve.times.shape[0])
    light_curve.times = np.roll(light_curve.times, random_index)
    light_curve.fluxes = np.roll(light_curve.fluxes, random_index)
    return light_curve
