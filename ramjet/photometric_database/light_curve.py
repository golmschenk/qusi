"""
Code for a class to represent a light curve. See the contained class docstring for more details.
"""
from typing import Union, Dict

import numpy as np
import pandas as pd


class LightCurve:
    """
    A class to represent a light curve. A light curve is a collection of data which may includes times, fluxes,
    flux errors, and related values.
    """
    def __init__(self):
        self.data_frame: pd.DataFrame = pd.DataFrame()
        self.flux_column_name: Union[str, None] = None
        self.time_column_name: Union[str, None] = None

    @property
    def fluxes(self) -> np.ndarray:
        """
        The fluxes of the light curve.

        :return: The fluxes.
        """
        return self.data_frame[self.flux_column_name].values

    @fluxes.setter
    def fluxes(self, value: np.ndarray):
        if self.flux_column_name is None:
            self.flux_column_name = 'flux'
        self.data_frame[self.flux_column_name] = value

    @property
    def times(self) -> np.ndarray:
        """
        The times of the light curve.

        :return: The times.
        """
        return self.data_frame[self.time_column_name].values

    @times.setter
    def times(self, value: np.ndarray):
        if self.time_column_name is None:
            self.time_column_name = 'time'
        self.data_frame[self.time_column_name] = value
