from typing import Tuple

import torch
import numpy.typing as npt
import numpy as np
from torch import Tensor

from qusi.light_curve_observation import LightCurveObservation


def from_light_curve_observation_to_fluxes_array_and_label_array(light_curve_observation: LightCurveObservation
                                               ) -> (npt.NDArray[np.float32], npt.NDArray[np.float32]):
    fluxes = light_curve_observation.light_curve.fluxes
    label = light_curve_observation.label
    return fluxes, np.array(label, dtype=np.float32)

def pair_array_to_tensor(arrays: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]) -> (Tensor, Tensor):
    return torch.tensor(arrays[0]), torch.tensor(arrays[1])
