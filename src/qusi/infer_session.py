
import numpy as np
import torch
from torch.nn import Module
from torch.types import Device
from torch.utils.data import DataLoader

from qusi.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset


def infer_session(infer_datasets: list[FiniteStandardLightCurveDataset], model: Module,
                  batch_size: int, device: Device):
    infer_dataloaders: list[DataLoader] = []
    for infer_dataset in infer_datasets:
        infer_dataloaders.append(DataLoader(infer_dataset, batch_size=batch_size, pin_memory=True))
    model.eval()
    results = []
    for infer_dataloader in infer_dataloaders:
        result = infer_phase(infer_dataloader, model, device=device)
        results.append(result)
    return results


def get_device() -> Device:
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
