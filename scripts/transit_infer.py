from pathlib import Path

import numpy as np
import torch
from bokeh.io import show
from bokeh.models import Div, Column
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.utils.data import DataLoader
from bokeh.plotting import figure as Figure

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

    model = Hadryss()
    train_session = InferSession.new(infer_datasets=[test_light_curve_dataset], model=model, batch_size=100)
    train_session.run()


def infer_session(dataset_path: Path):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    evaluation_dataset = NicerDataset.new(
        dataset_path=dataset_path,
        parameters_transform=PrecomputedNormalizeParameters(),
        phase_amplitudes_transform=PrecomputedNormalizePhaseAmplitudes())
    validation_dataset, test_dataset = split_dataset_into_fractional_datasets(evaluation_dataset, [0.5, 0.5])

    batch_size = 100

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = DataParallel(LiraTraditionalShape8xWidthWithNoDoNoBn())
    model = model.to(device)
    model.load_state_dict(torch.load('sessions/ncy8keio_latest_model.pt', map_location=device))
    model.eval()
    loss_function = PlusOneChiSquaredStatisticMetric()

    test_phase(test_dataloader, model, loss_function, device=device)


def test_phase(dataloader, model_: Module, loss_fn, device):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch_index, (inputs_tensor, targets) in enumerate(dataloader):
            print(batch_index * dataloader.batch_size)
            inputs_tensor = inputs_tensor.to(device)
            targets = targets
            predicted_targets = model_(inputs_tensor)
            test_loss += loss_fn(predicted_targets.to('cpu'), targets).to(device).item()
            pass


    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    main()

