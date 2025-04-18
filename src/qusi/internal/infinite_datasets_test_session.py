from torch.nn import Module
from torch.types import Device
from torch.utils.data import DataLoader
import torch

from qusi.internal.light_curve_dataset import LightCurveDataset
from qusi.internal.train_session import update_logging_metrics, get_cycle_metric_values


def infinite_datasets_test_session(test_datasets: list[LightCurveDataset], model: Module,
                                   metric_functions: list[Module], *, batch_size: int, device: Device, steps: int):
    """
    Runs a test session on finite datasets.

    :param test_datasets: A list of datasets to run the test session on.
    :param model: A model to perform the inference.
    :param metric_functions: A metrics to test.
    :param batch_size: A batch size to use during testing.
    :param device: A device to run the model on.
    :param steps: The number of steps to run on the infinite datasets.
    :return: A list of arrays, with one array for each test dataset, with each array containing an element for each
             metric that was tested.
    """
    test_dataloaders: list[DataLoader] = []
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=batch_size, pin_memory=True))
    model.eval()
    results = []
    for test_dataloader in test_dataloaders:
        result = infinite_dataset_test_phase(test_dataloader, model, metric_functions, device=device, steps=steps)
        results.append(result)
    return results


def infinite_dataset_test_phase(dataloader, model: Module, metric_functions: list[Module], device: Device, steps: int):
    batch_count = 0
    metric_totals = torch.zeros(size=[len(metric_functions)])
    model.eval()
    with torch.no_grad():
        for input_features, targets in dataloader:
            input_features = input_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predicted_targets = model(input_features)
            update_logging_metrics(predicted_targets, targets, metric_functions, metric_totals)
            batch_count += 1
            if batch_count >= steps:
                break
    cycle_metric_values = get_cycle_metric_values(metric_functions, metric_totals, batch_count)
    return cycle_metric_values
