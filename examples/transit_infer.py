from pathlib import Path

import numpy as np
import torch

from qusi.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset
from qusi.hadryss_model import Hadryss
from qusi.infer_session import get_device, infer_session
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

    model = Hadryss.new()
    device = get_device()
    model.load_state_dict(torch.load('sessions/<wandb_run_name>_latest_model.pt', map_location=device))
    confidences = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                                batch_size=100, device=device)[0]
    paths = list(get_infer_paths())
    paths_with_confidences = zip(paths, confidences)
    sorted_paths_with_confidences = sorted(
        paths_with_confidences, key=lambda path_with_confidence: path_with_confidence[1], reverse=True)
    print(sorted_paths_with_confidences)


if __name__ == '__main__':
    main()
