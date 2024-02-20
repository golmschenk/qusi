from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.light_curve_collection import LightCurveCollection

if TYPE_CHECKING:
    from collections.abc import Iterable


class ToyLightCurveCollection(LightCurveCollection):
    """
    A collection of simple toy light curves.
    """

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        return [Path("")]


class ToyFlatLightCurveCollection(ToyLightCurveCollection):
    def __init__(self):
        super().__init__()
        self.label = 0

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):  # noqa ARG002
        light_curve = ToyLightCurve.flat()
        return light_curve.times, light_curve.fluxes


class ToyFlatAtValueLightCurveCollection(ToyLightCurveCollection):
    def __init__(self):
        super().__init__()

    def get_paths(self) -> Iterable[Path]:
        paths = [Path(f"{index}") for index in range(10)]
        return paths

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        light_curve = ToyLightCurve.flat(float(path.name))
        return light_curve.times, light_curve.fluxes

    def load_label_from_path(self, path: Path) -> float | np.ndarray:
        label = float(path.name)
        return label


class ToySineWaveLightCurveCollection(ToyLightCurveCollection):
    def __init__(self):
        super().__init__()
        self.label = 1

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):  # noqa ARG002
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
        return LightCurve.from_times_and_fluxes(times=times, fluxes=fluxes)

    @classmethod
    def sine_wave(cls, period=50) -> LightCurve:
        """
        Creates a sine wave light curve.
        """
        length = 100
        periods_to_produce = length / period
        fluxes = (np.sin(np.linspace(0, np.pi * periods_to_produce, num=length, endpoint=False)) / 2) + 1
        times = np.arange(length)
        return LightCurve.from_times_and_fluxes(times=times, fluxes=fluxes)
