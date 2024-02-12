
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from qusi.light_curve_observation import LightCurveObservation


def from_light_curve_observation_to_fluxes_array_and_label_array(light_curve_observation: LightCurveObservation
                                                                 ) -> (
npt.NDArray[np.float32], npt.NDArray[np.float32]):
    fluxes = light_curve_observation.light_curve.fluxes
    label = light_curve_observation.label
    return fluxes, np.array(label, dtype=np.float32)


def pair_array_to_tensor(arrays: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]) -> (Tensor, Tensor):
    return torch.tensor(arrays[0]), torch.tensor(arrays[1])


def randomly_roll_elements(example: np.ndarray) -> np.ndarray:
    """Randomly rolls the elements."""
    example = np.roll(example, np.random.randint(example.shape[0]), axis=0)
    return example
