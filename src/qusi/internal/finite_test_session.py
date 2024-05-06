import torch
from torch.nn import Module
from torch.types import Device
from torch.utils.data import DataLoader

from qusi.internal.finite_standard_light_curve_observation_dataset import FiniteStandardLightCurveObservationDataset


def finite_datasets_test_session(
        test_datasets: list[FiniteStandardLightCurveObservationDataset],
        model: Module,
        metric_functions: list[Module],
        *,
        batch_size: int = 100,
        device: Device = torch.device('cpu'),
):
    """
    Runs a test session on finite datasets.

    :param test_datasets: A list of datasets to run the test session on.
    :param model: A model to perform the inference.
    :param metric_functions: A metrics to test.
    :param batch_size: A batch size to use during testing.
    :param device: A device to run the model on.
    :return: A list of arrays, with one array for each test dataset, with each array containing an element for each
             metric that was tested.
    """
    test_dataloaders: list[DataLoader] = []
    for test_dataset in test_datasets:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        test_dataloaders.append(test_dataloader)
    model.eval()
    results = []
    for test_dataloader in test_dataloaders:
        result = finite_dataset_test_phase(test_dataloader, model, metric_functions, device=device)
        results.append(result)
    return results


def finite_dataset_test_phase(dataloader, model: Module, metric_functions: list[Module], device: Device):
    batch_count = 0
    metric_totals = torch.zeros(size=[len(metric_functions)])
    model.eval()
    with torch.no_grad():
        for input_features, targets in dataloader:
            input_features_on_device = input_features.to(device, non_blocking=True)
            targets_on_device = targets.to(device, non_blocking=True)
            predicted_targets = model(input_features_on_device)
            for metric_function_index, metric_function in enumerate(metric_functions):
                batch_metric_value = metric_function(predicted_targets.to(device, non_blocking=True), targets_on_device)
                metric_totals[metric_function_index] += batch_metric_value.to("cpu", non_blocking=True)
            batch_count += 1
    cycle_metric_values = metric_totals / batch_count
    return cycle_metric_values
