from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt


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
