from pathlib import Path

import numpy as np

from qusi.light_curve import LightCurve
from qusi.light_curve_collection import LabeledLightCurveCollection, create_constant_label_for_path_function


class ToyLightCurve:
    """
    Simple toy light curves.
    """

    @classmethod
    def flat(cls, value: float = 1) -> LightCurve:
        """
        Creates a flat light curve.

        :param value: The flux value of the flat light curve.
        """
        length = 100
        fluxes = np.full(shape=[length], fill_value=value, dtype=np.float32)
        times = np.arange(length, dtype=np.float32)
        return LightCurve.new(times=times, fluxes=fluxes)

    @classmethod
    def sine_wave(cls, period=50) -> LightCurve:
        """
        Creates a sine wave light curve.
        """
        length = 100
        periods_to_produce = length / period
        fluxes = (np.sin(np.linspace(0, np.pi * periods_to_produce, num=length, endpoint=False)) / 2) + 1
        times = np.arange(length)
        return LightCurve.new(times=times, fluxes=fluxes)


def toy_light_curve_get_paths_function() -> list[Path]:
    """
    A fake function to fulfill the need of returning a list of paths.
    """
    return [Path('')]


def toy_flat_light_curve_load_times_and_fluxes(_path: Path) -> (np.ndarray, np.ndarray):
    """
    Loads a flat toy light curve.
    """
    light_curve = ToyLightCurve.flat()
    return light_curve.times, light_curve.fluxes


def toy_sine_wave_light_curve_load_times_and_fluxes(_path: Path) -> (np.ndarray, np.ndarray):
    """
    Loads a sine wave toy light curve.
    """
    light_curve = ToyLightCurve.sine_wave()
    return light_curve.times, light_curve.fluxes


toy_flat_light_curve_collection = LabeledLightCurveCollection.new(
    get_paths_function=toy_light_curve_get_paths_function,
    load_times_and_fluxes_from_path_function=toy_flat_light_curve_load_times_and_fluxes,
    load_label_from_path_function=create_constant_label_for_path_function(0)
)

toy_sine_wave_light_curve_collection = LabeledLightCurveCollection.new(
    get_paths_function=toy_light_curve_get_paths_function,
    load_times_and_fluxes_from_path_function=toy_sine_wave_light_curve_load_times_and_fluxes,
    load_label_from_path_function=create_constant_label_for_path_function(1)
)
