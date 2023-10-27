from pathlib import Path

import numpy as np

from qusi.hadryss_model import Hadryss
from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_dataset import LightCurveDataset
from qusi.train_session import TrainSession
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve


def get_negative_train_paths():
    return list(Path('data/spoc_transit_experiment/train/negatives').glob('*.fits'))


def get_negative_validation_paths():
    return list(Path('data/spoc_transit_experiment/validation/negatives').glob('*.fits'))


def get_positive_train_paths():
    return list(Path('data/spoc_transit_experiment/train/positives').glob('*.fits'))


def get_positive_validation_paths():
    return list(Path('data/spoc_transit_experiment/validation/positives').glob('*.fits'))


def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def positive_label_function(_path: Path) -> int:
    return 1


def negative_label_function(_path: Path) -> int:
    return 0


def main():
    positive_train_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_positive_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_train_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_negative_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    positive_validation_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_positive_validation_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_validation_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_negative_validation_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)

    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_train_light_curve_collection,
                                          negative_train_light_curve_collection])
    validation_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_validation_light_curve_collection,
                                          negative_validation_light_curve_collection])
    model = Hadryss()
    train_session = TrainSession.new(train_datasets=[train_light_curve_dataset],
                                     validation_datasets=[validation_light_curve_dataset],
                                     model=model, batch_size=100, cycles=100, train_steps_per_cycle=100,
                                     validation_steps_per_cycle=10)
    train_session.run()


if __name__ == '__main__':
    main()
