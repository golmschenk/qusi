from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from qusi.internal.light_curve_observation import LightCurveObservation


def from_light_curve_observation_to_fluxes_array_and_label_array(
    light_curve_observation: LightCurveObservation,
) -> (npt.NDArray[np.float32], npt.NDArray[np.float32]):
    """
    Extracts the fluxes and label from a light curve observation.

    :param light_curve_observation: The light curve observation.
    :return: The fluxes and label array.
    """
    fluxes = light_curve_observation.light_curve.fluxes
    label = light_curve_observation.label
    return fluxes, np.array(label, dtype=np.float32)


def pair_array_to_tensor(
    arrays: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
) -> (Tensor, Tensor):
    """
    Converts a pair of arrays to a pair of tensors.

    :param arrays: The arrays to convert.
    :return: The tensors.
    """
    return torch.tensor(arrays[0], dtype=torch.float32), torch.tensor(
        arrays[1], dtype=torch.float32
    )


def randomly_roll_elements(example: np.ndarray) -> np.ndarray:
    """Randomly rolls the elements."""
    example = np.roll(example, np.random.randint(example.shape[0]), axis=0)
    return example


def normalize_tensor_by_modified_z_score(tensor: Tensor) -> Tensor:
    """
    Normalizes a tensor by a modified z-score. That is, normalizes the values of the tensor based on the median
    absolute deviation.

    :param tensor: The tensor to normalize.
    :return: The normalized tensor.
    """
    median = torch.median(tensor)
    deviation_from_median = tensor - median
    absolute_deviation_from_median = torch.abs(deviation_from_median)
    median_absolute_deviation_from_median = torch.median(absolute_deviation_from_median)
    if median_absolute_deviation_from_median != 0:
        modified_z_score = (
                0.6745 * deviation_from_median / median_absolute_deviation_from_median
        )
    else:
        modified_z_score = torch.zeros_like(tensor)
    return modified_z_score


def make_uniform_length(example: np.ndarray, length: int) -> np.ndarray:
    """Makes the example a specific length, by clipping those too large and repeating those too small."""
    if len(example.shape) not in [1, 2]:  # Only tested for 1D and 2D cases.
        raise ValueError(
            f"Light curve dimensions expected to be in [1, 2], but found {len(example.shape)}"
        )
    if example.shape[0] == length:
        pass
    elif example.shape[0] > length:
        example = example[:length]
    else:
        elements_to_repeat = length - example.shape[0]
        if len(example.shape) == 1:
            example = np.pad(example, (0, elements_to_repeat), mode="wrap")
        else:
            example = np.pad(example, ((0, elements_to_repeat), (0, 0)), mode="wrap")
    return example


def remove_random_elements(array: np.ndarray, ratio: float = 0.01) -> np.ndarray:
    """Removes random values from an array."""
    light_curve_length = array.shape[0]
    max_values_to_remove = int(light_curve_length * ratio)
    if max_values_to_remove != 0:
        values_to_remove = np.random.randint(max_values_to_remove)
    else:
        values_to_remove = 0
    random_indexes = np.random.choice(range(light_curve_length), values_to_remove, replace=False)
    return np.delete(array, random_indexes, axis=0)