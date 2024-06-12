import math

import random
from functools import partial

from pathlib import Path

import numpy as np
from scipy import signal

from qusi.internal.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset
from qusi.internal.light_curve import LightCurve
from qusi.internal.light_curve_collection import (
    LightCurveObservationCollection,
    create_constant_label_for_path_function, LightCurveCollection,
)
from qusi.internal.light_curve_dataset import LightCurveDataset, \
    default_light_curve_observation_post_injection_transform


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
        fluxes = (
            np.sin(
                np.linspace(0, np.pi * periods_to_produce, num=length, endpoint=False)
            )
            / 2
        ) + 1
        times = np.arange(length)
        return LightCurve.new(times=times, fluxes=fluxes)


def toy_light_curve_get_paths_function() -> list[Path]:
    """
    A fake function to fulfill the need of returning a list of paths.
    """
    return [Path("")]


def toy_flat_light_curve_load_times_and_fluxes(_path: Path) -> (np.ndarray, np.ndarray):
    """
    Loads a flat toy light curve.
    """
    light_curve = ToyLightCurve.flat()
    return light_curve.times, light_curve.fluxes


def toy_sine_wave_light_curve_load_times_and_fluxes(
        _path: Path,
) -> (np.ndarray, np.ndarray):
    """
    Loads a sine wave toy light curve.
    """
    light_curve = ToyLightCurve.sine_wave()
    return light_curve.times, light_curve.fluxes


def get_toy_flat_light_curve_observation_collection() -> LightCurveObservationCollection:
    return LightCurveObservationCollection.new(
        get_paths_function=toy_light_curve_get_paths_function,
        load_times_and_fluxes_from_path_function=toy_flat_light_curve_load_times_and_fluxes,
        load_label_from_path_function=create_constant_label_for_path_function(0),
    )


def get_toy_sine_wave_light_curve_observation_collection() -> LightCurveObservationCollection:
    return LightCurveObservationCollection.new(
        get_paths_function=toy_light_curve_get_paths_function,
        load_times_and_fluxes_from_path_function=toy_sine_wave_light_curve_load_times_and_fluxes,
        load_label_from_path_function=create_constant_label_for_path_function(1),
    )


def get_toy_flat_light_curve_collection() -> LightCurveCollection:
    return LightCurveCollection.new(
        get_paths_function=toy_light_curve_get_paths_function,
        load_times_and_fluxes_from_path_function=toy_flat_light_curve_load_times_and_fluxes,
    )


def get_toy_sine_wave_light_curve_collection() -> LightCurveCollection:
    return LightCurveCollection.new(
        get_paths_function=toy_light_curve_get_paths_function,
        load_times_and_fluxes_from_path_function=toy_sine_wave_light_curve_load_times_and_fluxes,
    )


def get_toy_dataset():
    return LightCurveDataset.new(
        standard_light_curve_collections=[
            get_toy_sine_wave_light_curve_observation_collection(),
            get_toy_flat_light_curve_observation_collection(),
        ]
    )


def get_toy_finite_light_curve_dataset() -> FiniteStandardLightCurveDataset:
    return FiniteStandardLightCurveDataset.new(
        light_curve_collections=[
            get_toy_sine_wave_light_curve_collection(),
            get_toy_flat_light_curve_collection(),
        ]
    )


def get_square_wave_light_curve_observation_collection() -> LightCurveObservationCollection:
    return LightCurveObservationCollection.new(
        get_paths_function=toy_light_curve_get_paths_function,
        load_times_and_fluxes_from_path_function=square_wave_light_curve_load_times_and_fluxes,
        load_label_from_path_function=create_constant_label_for_path_function(2),
    )


square_wave_random_generator = random.Random()


def square_wave_light_curve_load_times_and_fluxes(_path: Path) -> (np.ndarray, np.ndarray):
    """
    Loads a square wave light curve.
    """
    length = 100
    number_of_cycles = square_wave_random_generator.random() + 1 * 9
    linear_space = np.linspace(0, 1, length, endpoint=False)
    phases = math.tau * number_of_cycles * linear_space
    times = np.arange(length, dtype=np.float32)
    fluxes = signal.square(phases)
    return times, fluxes


def get_toy_multi_class_light_curve_dataset() -> LightCurveDataset:
    return LightCurveDataset.new(
        standard_light_curve_collections=[
            get_toy_flat_light_curve_observation_collection(),
            get_toy_sine_wave_light_curve_observation_collection(),
            get_square_wave_light_curve_observation_collection(),
        ],
        post_injection_transform=partial(default_light_curve_observation_post_injection_transform,
                                         length=100, number_of_classes=3, randomize=False)
    )
