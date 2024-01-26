from pathlib import Path

import numpy as np
import torch

from qusi.finite_standard_light_curve_dataset import FiniteStandardLightCurveDataset
from qusi.hadryss_model import Hadryss
from qusi.infer_session import infer_session, get_device
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
    model.load_state_dict(torch.load('sessions/pleasant-lion-32_latest_model.pt', map_location=device))
    results = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                            batch_size=100, device=device)
    return results


if __name__ == '__main__':
    main()
