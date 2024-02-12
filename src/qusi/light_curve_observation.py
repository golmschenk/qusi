from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Self

from qusi.light_curve import LightCurve, randomly_roll_light_curve, remove_nan_flux_data_points_from_light_curve


@dataclass
class LightCurveObservation:
    """
    An observation containing a light curve and label. Note, this is an observation in machine learning terms, not to be
    confused with an astrophysical observation.

    :ivar light_curve: The light curve.
    :ivar label: The integer classification label.
    """
    light_curve: LightCurve
    label: int

    @classmethod
    def new(cls, light_curve: LightCurve, label: int) -> Self:
        """
        Creates a new LightCurveObservation.

        :param light_curve: The light curve.
        :param label: The integer classification label.
        :return: The observation.
        """
        return cls(light_curve=light_curve, label=label)


def remove_nan_flux_data_points_from_light_curve_observation(
        light_curve_observation: LightCurveObservation) -> LightCurveObservation:
    light_curve_observation = deepcopy(light_curve_observation)
    light_curve_observation.light_curve = remove_nan_flux_data_points_from_light_curve(
        light_curve_observation.light_curve)
    return light_curve_observation


def randomly_roll_light_curve_observation(
        light_curve_observation: LightCurveObservation) -> LightCurveObservation:
    light_curve_observation = deepcopy(light_curve_observation)
    light_curve_observation.light_curve = randomly_roll_light_curve(
        light_curve_observation.light_curve)
    return light_curve_observation
