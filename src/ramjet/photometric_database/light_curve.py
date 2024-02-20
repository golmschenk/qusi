"""
Code for a class to represent a light curve. See the contained class docstring for more details.
"""
from __future__ import annotations

import lightkurve.lightcurve
import numpy as np
import pandas as pd
from lightkurve.periodogram import LombScarglePeriodogram


class LightCurve:
    """
    A class to represent a light curve. A light curve is a collection of data which may includes times, fluxes,
    flux errors, and related values.
    """

    def __init__(self):
        self.data_frame: pd.DataFrame = pd.DataFrame()
        self.flux_column_names: list[str] = []
        self.time_column_name: str | None = None
        self._variability_period: float | None = None
        self._variability_period_epoch: float | None = None
        self.folded_times_column_name = "_folded_times"

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
            default_flux_name = "flux"
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
            self.time_column_name = "time"
        self.data_frame[self.time_column_name] = value

    @property
    def folded_times(self):
        if self.folded_times_column_name not in self.data_frame.columns:
            error_message = "Light curve has not been folded."
            raise MissingFoldedTimesError(error_message)
        return self.data_frame[self.folded_times_column_name].values

    @folded_times.setter
    def folded_times(self, value: np.ndarray):
        self.data_frame[self.folded_times_column_name] = value

    @property
    def variability_period(self) -> float:
        if self._variability_period is None:
            self.get_variability_phase_folding_parameters()
        return self._variability_period

    @property
    def variability_period_epoch(self) -> float:
        if self._variability_period_epoch is None:
            self.get_variability_phase_folding_parameters()
        return self._variability_period_epoch

    def convert_column_to_relative_scale(self, column_name: str):
        """
        Converts a column to relative scale.

        :param column_name: The name of the column to be converted.
        """
        self.data_frame[column_name] = self.data_frame[column_name] / np.nanmedian(self.data_frame[column_name])

    def convert_columns_to_relative_scale(self, column_names: list[str]):
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

    def get_variability_phase_folding_parameters(
        self, minimum_period: float | None = None, maximum_period: float | None = None
    ) -> (float, float, float, float, float):
        (
            fold_period,
            fold_epoch,
            time_bin_size,
            minimum_bin_phase,
            maximum_bin_phase,
            inlier_lightkurve_light_curve,
            periodogram,
            folded_lightkurve_light_curve,
        ) = self.get_variability_phase_folding_parameters_and_folding_lightkurve_light_curves(
            minimum_period=minimum_period, maximum_period=maximum_period
        )
        self._variability_period = fold_period
        self._variability_period_epoch = fold_epoch
        return fold_period, fold_epoch, time_bin_size, minimum_bin_phase, maximum_bin_phase

    def get_variability_phase_folding_parameters_and_folding_lightkurve_light_curves(
        self, minimum_period: float | None = None, maximum_period: float | None = None
    ):
        np.median(np.diff(self.times[~np.isnan(self.times)]))
        lightkurve_light_curve = self.to_lightkurve()
        inlier_lightkurve_light_curve = lightkurve_light_curve.remove_outliers(sigma=3)
        periodogram = LombScarglePeriodogram.from_lightcurve(
            inlier_lightkurve_light_curve,
            oversample_factor=100,
            minimum_period=minimum_period,
            maximum_period=maximum_period,
        )
        folded_lightkurve_light_curve = inlier_lightkurve_light_curve.fold(
            period=periodogram.period_at_max_power, wrap_phase=periodogram.period_at_max_power
        )
        fold_period = folded_lightkurve_light_curve.period.value
        time_bin_size = fold_period / 25
        binned_folded_lightkurve_light_curve = folded_lightkurve_light_curve.bin(
            time_bin_size=time_bin_size, aggregate_func=np.nanmedian
        )
        minimum_bin_index = np.nanargmin(binned_folded_lightkurve_light_curve.flux.value)
        maximum_bin_index = np.nanargmax(binned_folded_lightkurve_light_curve.flux.value)
        minimum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[minimum_bin_index]
        maximum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[maximum_bin_index]
        fold_epoch = inlier_lightkurve_light_curve.time.value[0]
        return (
            fold_period,
            fold_epoch,
            time_bin_size,
            minimum_bin_phase,
            maximum_bin_phase,
            inlier_lightkurve_light_curve,
            periodogram,
            folded_lightkurve_light_curve,
        )

    def fold(self, period: float, epoch: float) -> None:
        self.folded_times = (self.times - epoch) % period


class MissingFoldedTimesError(Exception):
    pass
