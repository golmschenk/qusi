from typing import Self

import numpy as np
import numpy.typing as npt


class LightCurve:
    """
    A class to represent a light curve. A light curve is a collection of data which may includes times, fluxes,
    and related values.

    :ivar times: The times of the light curve.
    :ivar fluxes: The fluxes of the light curve.
    """
    def __init__(self, times: npt.NDArray[np.float32], fluxes: npt.NDArray[np.float32]):
        self.times: npt.NDArray[np.float32] = times
        self.fluxes: npt.NDArray[np.float32] = fluxes

    @classmethod
    def new(cls, times: npt.NDArray[np.float32], fluxes: npt.NDArray[np.float32]) -> Self:
        """
        Creates a new light curve.

        :param times: The times of the light curve.
        :param fluxes: The fluxes of the light curve.
        :return: The light curve.
        """
        return cls(times=times, fluxes=fluxes)
