import logging

import numpy as np
import torch
from torch.nn import Module
from torch.types import Device
from torch.utils.data import DataLoader

from qusi.internal.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset


logger = logging.getLogger(__name__)


def infer_session(
        infer_datasets: list[FiniteStandardLightCurveDataset],
        model: Module,
        *,
        batch_size: int,
        device: Device,
        workers_per_dataloader: int = 0,
) -> list[np.ndarray]:
    """
    Runs an infer session on finite datasets.

    :param infer_datasets: The list of datasets to run the infer session on.
    :param model: The model to perform the inference.
    :param batch_size: The batch size to use during inference.
    :param device: The device to run the model on.
    :return: A list of arrays with each element being the array predicted for each light curve in the dataset.
    """
    logger.info(f'Creating dataloader workers...')
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    if workers_per_dataloader == 0:
        prefetch_factor = None
    else:
        prefetch_factor = 10
    infer_dataloaders: list[DataLoader] = []
    for infer_dataset in infer_datasets:
        infer_dataloader = DataLoader(infer_dataset, batch_size=batch_size, pin_memory=True,
                                      prefetch_factor=prefetch_factor, num_workers=workers_per_dataloader)
        infer_dataloaders.append(infer_dataloader)
    model.eval()
    results = []
    logger.info(f'Entering infer loop...')
    for infer_dataloader in infer_dataloaders:
        result = infer_phase(infer_dataloader, model, device=device)
        results.append(result)
    return results


def infer_phase(dataloader, model: Module, device: Device):
    batch_count = 0
    processed_count = 0
    batches_of_predicted_targets = []
    model = model.to(device=device)
    model.eval()
    with torch.no_grad():
        for input_features in dataloader:
            input_features_on_device = input_features.to(device, non_blocking=True)
            batch_predicted_targets = model(input_features_on_device)
            batch_predicted_targets_array = batch_predicted_targets.cpu().numpy()
            batches_of_predicted_targets.append(batch_predicted_targets_array)
            batch_count += 1
            processed_count += batch_predicted_targets_array.shape[0]
            logger.info(f'Processed: {processed_count}.')
    predicted_targets = np.concatenate(batches_of_predicted_targets, axis=0)
    return predicted_targets
