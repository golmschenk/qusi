import torch
import numpy.typing as npt
import numpy as np
from torch import Tensor

from qusi.light_curve_observation import LightCurveObservation


def from_observation_to_fluxes_array_and_label_array(light_curve_observation: LightCurveObservation
                                               ) -> (npt.NDArray[np.float32], npt.NDArray[np.float32]):
    fluxes = light_curve_observation.light_curve.fluxes
    label = light_curve_observation.label
    return fluxes, np.array(label, dtype=np.float32)


def to_tensor(array: npt.NDArray[np.float32]) -> Tensor:
    return torch.tensor(array)
