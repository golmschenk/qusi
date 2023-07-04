from pathlib import Path
from typing import Iterable, Union

import numpy as np

from qusi.light_curve import LightCurve
from qusi.light_curve_collection import LabeledLightCurveCollection, create_constant_label_for_path_function


class ToyLightCurveCollection(LabeledLightCurveCollection):
    """
    A collection of simple toy light curves.
    """
    def __init__(self, load_times_and_fluxes_from_path_function, load_label_from_path_function):
        super().__init__(get_paths_function=self.get_paths,
                         load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path_function,
                         load_label_from_path_function=load_label_from_path_function)

    @staticmethod
    def get_paths() -> Iterable[Path]:
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        return [Path('')]


class ToyFlatLightCurveCollection(ToyLightCurveCollection):
    def __init__(self):
        super().__init__(load_times_and_fluxes_from_path_function=self.load_times_and_fluxes_from_path,
                         load_label_from_path_function=create_constant_label_for_path_function(0))

    @staticmethod
    def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
        light_curve = ToyLightCurve.flat()
        return light_curve.times, light_curve.fluxes


class ToyFlatAtValueLightCurveCollection(ToyLightCurveCollection):
    def __init__(self):
        super().__init__(load_times_and_fluxes_from_path_function=self.load_times_and_fluxes_from_path,
                         load_label_from_path_function=self.load_label_from_path)

    @staticmethod
    def get_paths() -> Iterable[Path]:
        paths = [Path(f'{index}') for index in range(10)]
        return paths

    @staticmethod
    def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
        light_curve = ToyLightCurve.flat(float(path.name))
        return light_curve.times, light_curve.fluxes

    @staticmethod
    def load_label_from_path(path: Path) -> int:
        label = int(path.name)
        return label


class ToySineWaveLightCurveCollection(ToyLightCurveCollection):
    def __init__(self):
        super().__init__(load_times_and_fluxes_from_path_function=self.load_times_and_fluxes_from_path,
                         load_label_from_path_function=create_constant_label_for_path_function(1))

    @staticmethod
    def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
        light_curve = ToyLightCurve.sine_wave()
        return light_curve.times, light_curve.fluxes


class ToyLightCurve:
    """
    Simple toy light curves.
    """

    @classmethod
    def flat(cls, value: float = 1) -> LightCurve:
        """
        Creates a flat light curve.
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
