"""
Code for a class to represent a light curve. See the contained class docstring for more details.
"""
from typing import Union, Dict, List

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

    def convert_column_to_relative_scale(self, column_name: str):
        """
        Converts a column to relative scale.

        :param column_name: The name of the column to be converted.
        """
        self.data_frame[column_name] = self.data_frame[column_name] / np.median(self.data_frame[column_name])

    def convert_columns_to_relative_scale(self, column_names: List[str]):
        """
        Converts multiple columns to relative scale.

        :param column_names: The list of names of columns to be converted.
        """
        for column_name in column_names:
            self.convert_column_to_relative_scale(column_name)
