"""
Code for a class to represent a light curve. See the contained class docstring for more details.
"""
from __future__ import annotations

from abc import ABC
from typing import Union, List

import lightkurve.lightcurve
import numpy as np
import pandas as pd
from lightkurve.periodogram import LombScarglePeriodogram
import scipy.signal


class LightCurve(ABC):
    """
    A class to represent a light curve. A light curve is a collection of data which may includes times, fluxes,
    flux errors, and related values.
    """
    def __init__(self):
        self.data_frame: pd.DataFrame = pd.DataFrame()
        self.flux_column_names: List[str] = []
        self.time_column_name: Union[str, None] = None

    @property
    def fluxes(self) -> np.ndarray:
        """
        The fluxes of the light curve.

        :return: The fluxes.
        """
        return self.data_frame[self.flux_column_names[0]].values

    @fluxes.setter
    def fluxes(self, value: np.ndarray):
        if len(self.flux_column_names) == 0:
            default_flux_name = 'flux'
            self.flux_column_names.append(default_flux_name)
        self.data_frame[self.flux_column_names[0]] = value

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
        self.data_frame[column_name] = self.data_frame[column_name] / np.nanmedian(self.data_frame[column_name])

    def convert_columns_to_relative_scale(self, column_names: List[str]):
        """
        Converts multiple columns to relative scale.

        :param column_names: The list of names of columns to be converted.
        """
        for column_name in column_names:
            self.convert_column_to_relative_scale(column_name)

    def convert_to_relative_scale(self):
        """
        Converts the light curve to relative scale.
        """
        self.convert_columns_to_relative_scale(self.flux_column_names)

    @classmethod
    def from_times_and_fluxes(cls, times: np.ndarray, fluxes: np.ndarray) -> LightCurve:
        light_curve = LightCurve()
        light_curve.times = times
        light_curve.fluxes = fluxes
        return light_curve

    def to_lightkurve(self) -> lightkurve.lightcurve.LightCurve:
        return lightkurve.lightcurve.LightCurve(time=self.times, flux=self.fluxes)

    def get_phase_folding_parameters(self) -> (float, float, float, float, float):
        median_time_step = np.median(np.diff(self.times[~np.isnan(self.times)]))
        time_bin_size = median_time_step
        lightkurve_light_curve = self.to_lightkurve()
        inlier_lightkurve_light_curve = lightkurve_light_curve.remove_outliers()
        periodogram = LombScarglePeriodogram.from_lightcurve(inlier_lightkurve_light_curve, oversample_factor=20)
        folded_lightkurve_light_curve = inlier_lightkurve_light_curve.fold(period=periodogram.period_at_max_power)
        binned_folded_lightkurve_light_curve = folded_lightkurve_light_curve.bin(time_bin_size=time_bin_size,
                                                                                 aggregate_func=np.nanmedian)
        minimum_bin_index = np.nanargmin(binned_folded_lightkurve_light_curve.flux.value)
        maximum_bin_index = np.nanargmax(binned_folded_lightkurve_light_curve.flux.value)
        minimum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[minimum_bin_index]
        maximum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[maximum_bin_index]
        fold_period = folded_lightkurve_light_curve.period.value
        fold_epoch = inlier_lightkurve_light_curve.time.value[0]
        return fold_period, fold_epoch, time_bin_size, minimum_bin_phase, maximum_bin_phase
