from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.nn import Module
from torch.types import Device
from torch.utils.data import DataLoader

from qusi.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset
from qusi.hadryss_model import Hadryss
from qusi.light_curve_collection import LightCurveCollection
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve


def get_infer_paths():
    return (list(Path('data/spoc_transit_experiment/test/negatives').glob('*.fits')) +
            list(Path('data/spoc_transit_experiment/test/positives').glob('*.fits')))


def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def main():
    infer_light_curve_collection = LightCurveCollection.new(
        get_paths_function=get_infer_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path)

    test_light_curve_dataset = FiniteStandardLightCurveDataset.new(
        light_curve_collections=[infer_light_curve_collection])

    model = Hadryss()
    device = get_device()
    model.load_state_dict(torch.load('sessions/pleasant-lion-32_latest_model.pt', map_location=device))
    results = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                            batch_size=100, device=device)
    return results


def infer_session(infer_datasets: List[FiniteStandardLightCurveDataset], model: Module,
                  batch_size: int, device: Device):
    infer_dataloaders: List[DataLoader] = []
    for infer_dataset in infer_datasets:
        infer_dataloaders.append(DataLoader(infer_dataset, batch_size=batch_size, pin_memory=True))
    model.eval()
    results = []
    for infer_dataloader in infer_dataloaders:
        result = infer_phase(infer_dataloader, model, device=device)
        results.append(result)
    return results


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def infer_phase(dataloader, model: Module, device: Device):
    batch_count = 0
    batches_of_predicted_targets = []
    model.eval()
    with torch.no_grad():
        for input_features in dataloader:
            input_features = input_features.to(device, non_blocking=True)
            batch_predicted_targets = model(input_features)
            batches_of_predicted_targets.append(batch_predicted_targets)
            batch_count += 1
    predicted_targets = np.concatenate(batches_of_predicted_targets, axis=0)
    return predicted_targets


if __name__ == '__main__':
    main()
