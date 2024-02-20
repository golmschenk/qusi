import math
from enum import Enum

import numpy as np
import scipy.stats
from numpy import typing as npt
from scipy.interpolate import interp1d


class OutOfBoundsInjectionHandlingMethod(Enum):
    """
    An enum of approaches for handling cases where the injectable signal is shorter than the injectee signal.
    """

    ERROR = "error"
    REPEAT_SIGNAL = "repeat_signal"
    RANDOM_INJECTION_LOCATION = "random_inject_location"


class BaselineFluxEstimationMethod(Enum):
    """
    An enum of to designate the type of baseline flux estimation method to use during training.
    """

    MEDIAN = "median"
    MEDIAN_ABSOLUTE_DEVIATION = "median_absolute_deviation"


def inject_signal_into_light_curve_with_intermediates(
    light_curve_times: npt.NDArray[np.float64],
    light_curve_fluxes: npt.NDArray[np.float64],
    signal_times: npt.NDArray[np.float64],
    signal_magnifications: npt.NDArray[np.float64],
    out_of_bounds_injection_handling_method: OutOfBoundsInjectionHandlingMethod = (
        OutOfBoundsInjectionHandlingMethod.ERROR
    ),
    baseline_flux_estimation_method: BaselineFluxEstimationMethod = BaselineFluxEstimationMethod.MEDIAN,
) -> (npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]):
    """
    Injects a synthetic magnification signal into real light curve fluxes.

    :param light_curve_times: The times of the flux observations of the light curve.
    :param light_curve_fluxes: The fluxes of the light curve to be injected into.
    :param signal_times: The times of the synthetic magnifications.
    :param signal_magnifications: The synthetic magnifications to inject.
    :param out_of_bounds_injection_handling_method: The method to use to handle out of bounds injection.
    :param baseline_flux_estimation_method: The method to use to estimate the baseline flux of the light curve
                                            for scaling the signal magnifications.
    :return: The fluxes with the injected signal, the offset signal times, and the signal flux.
    """
    minimum_light_curve_time = np.min(light_curve_times)
    relative_light_curve_times = light_curve_times - minimum_light_curve_time
    relative_signal_times = signal_times - np.min(signal_times)
    signal_time_length = np.max(relative_signal_times)
    light_curve_time_length = np.max(relative_light_curve_times)
    time_length_difference = light_curve_time_length - signal_time_length
    signal_start_offset = (np.random.random() * time_length_difference) + minimum_light_curve_time
    offset_signal_times = relative_signal_times + signal_start_offset
    if baseline_flux_estimation_method == BaselineFluxEstimationMethod.MEDIAN_ABSOLUTE_DEVIATION:
        baseline_flux = scipy.stats.median_abs_deviation(light_curve_fluxes)
        baseline_to_median_absolute_deviation_ratio = 10  # Arbitrarily chosen to give a reasonable scale.
        baseline_flux *= baseline_to_median_absolute_deviation_ratio
    else:
        baseline_flux = np.median(light_curve_fluxes)
    signal_fluxes = (signal_magnifications * baseline_flux) - baseline_flux
    if out_of_bounds_injection_handling_method is OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION:
        signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=False, fill_value=0)
    elif (
        out_of_bounds_injection_handling_method is OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        and time_length_difference > 0
    ):
        before_signal_gap = signal_start_offset - minimum_light_curve_time
        after_signal_gap = time_length_difference - before_signal_gap
        minimum_signal_time_step = np.min(np.diff(offset_signal_times))
        before_repeats_needed = math.ceil(before_signal_gap / (signal_time_length + minimum_signal_time_step))
        after_repeats_needed = math.ceil(after_signal_gap / (signal_time_length + minimum_signal_time_step))
        repeated_signal_fluxes = np.tile(signal_fluxes, before_repeats_needed + 1 + after_repeats_needed)
        repeated_signal_times = None
        for repeat_index in range(-before_repeats_needed, after_repeats_needed + 1):
            repeat_signal_start_offset = (signal_time_length + minimum_signal_time_step) * repeat_index
            if repeated_signal_times is None:
                repeated_signal_times = offset_signal_times + repeat_signal_start_offset
            else:
                repeat_index_signal_times = offset_signal_times + repeat_signal_start_offset
                repeated_signal_times = np.concatenate([repeated_signal_times, repeat_index_signal_times])
        signal_flux_interpolator = interp1d(repeated_signal_times, repeated_signal_fluxes, bounds_error=True)
    else:
        signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=True)
    interpolated_signal_fluxes = signal_flux_interpolator(light_curve_times)
    fluxes_with_injected_signal = light_curve_fluxes + interpolated_signal_fluxes
    return fluxes_with_injected_signal, offset_signal_times, signal_fluxes
