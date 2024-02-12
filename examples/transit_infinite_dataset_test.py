from pathlib import Path

import numpy as np
import torch
from torch.nn import BCELoss, Module
from torch.types import Device
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from qusi.hadryss_model import Hadryss
from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_dataset import LightCurveDataset
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve


def get_negative_test_paths():
    return list(Path('data/spoc_transit_experiment/test/negatives').glob('*.fits'))


def get_positive_test_paths():
    return list(Path('data/spoc_transit_experiment/test/positives').glob('*.fits'))


def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def positive_label_function(_path: Path) -> int:
    return 1


def negative_label_function(_path: Path) -> int:
    return 0


def main():
    positive_test_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_positive_test_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_test_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_negative_test_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)

    test_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_test_light_curve_collection,
                                          negative_test_light_curve_collection])

    model = Hadryss.new()
    device = get_device()
    model.load_state_dict(torch.load('sessions/pleasant-lion-32_latest_model.pt', map_location=device))
    metric_functions = [BinaryAccuracy(), BCELoss()]
    results = infinite_datasets_test_session(test_datasets=[test_light_curve_dataset], model=model,
                                             metric_functions=metric_functions, batch_size=100, device=device,
                                             steps=100)
    return results


def infinite_datasets_test_session(test_datasets: list[LightCurveDataset], model: Module,
                                   metric_functions: list[Module], batch_size: int, device: Device, steps: int):
    test_dataloaders: list[DataLoader] = []
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=batch_size, pin_memory=True))
    model.eval()
    results = []
    for test_dataloader in test_dataloaders:
        result = infinite_dataset_test_phase(test_dataloader, model, metric_functions, device=device, steps=steps)
        results.append(result)
    return results


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def infinite_dataset_test_phase(dataloader, model: Module, metric_functions: list[Module], device: Device, steps: int):
    batch_count = 0
    metric_totals = torch.zeros(size=[len(metric_functions)])
    model.eval()
    with torch.no_grad():
        for input_features, targets in dataloader:
            input_features = input_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predicted_targets = model(input_features)
            for metric_function_index, metric_function in enumerate(metric_functions):
                batch_metric_value = metric_function(predicted_targets.to(device, non_blocking=True),
                                                     targets)
                metric_totals[metric_function_index] += batch_metric_value.to('cpu', non_blocking=True)
            batch_count += 1
            if batch_count >= steps:
                break
    cycle_metric_values = metric_totals / batch_count
    return cycle_metric_values


if __name__ == '__main__':
    main()
