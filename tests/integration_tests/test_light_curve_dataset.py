from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from qusi.internal.light_curve_collection import LightCurveObservationCollection
from qusi.internal.light_curve_dataset import LightCurveDataset
from qusi.internal.light_curve_transforms import from_light_curve_observation_to_fluxes_array_and_label_array, \
    pair_array_to_tensor


def get_paths() -> list[Path]:
    return [Path('1'), Path('2'), Path('3'), Path('4'), Path('5'), Path('6'), Path('7'), Path('8')]

def load_times_and_fluxes_from_path(path: Path) -> [npt.NDArray, npt.NDArray]:
    value = float(str(path))
    return np.array([value]), np.array([value])

def load_label_from_path_function(path: Path) -> int:
    value = int(str(path))
    return value * 10

def post_injection_transform(x):
    x = from_light_curve_observation_to_fluxes_array_and_label_array(x)
    x = pair_array_to_tensor(x)
    return x


def test_multiple_data_workers_do_not_return_the_same_batches():
    light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=load_label_from_path_function)
    light_curve_dataset = LightCurveDataset.new(standard_light_curve_collections=[light_curve_collection],
                                                post_injection_transform=post_injection_transform)
    multi_worker_dataloader = DataLoader(light_curve_dataset, batch_size=4, num_workers=2, prefetch_factor=1)
    multi_worker_dataloader_iter = iter(multi_worker_dataloader)
    multi_worker_batch0 = next(multi_worker_dataloader_iter)[0].numpy()[:, 0]
    multi_worker_batch1 = next(multi_worker_dataloader_iter)[0].numpy()[:, 0]
    assert not np.array_equal(multi_worker_batch0, multi_worker_batch1)
