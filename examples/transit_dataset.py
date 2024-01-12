from pathlib import Path

from qusi.finite_standard_light_curve_observation_dataset import FiniteStandardLightCurveObservationDataset
from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_dataset import LightCurveDataset
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve


def get_positive_train_paths():
    return list(Path('data/spoc_transit_experiment/train/positives').glob('*.fits'))


def get_negative_train_paths():
    return list(Path('data/spoc_transit_experiment/train/negatives').glob('*.fits'))


def get_positive_validation_paths():
    return list(Path('data/spoc_transit_experiment/validation/positives').glob('*.fits'))


def get_negative_validation_paths():
    return list(Path('data/spoc_transit_experiment/validation/negatives').glob('*.fits'))


def get_negative_test_paths():
    return list(Path('data/spoc_transit_experiment/test/negatives').glob('*.fits'))


def get_positive_test_paths():
    return list(Path('data/spoc_transit_experiment/test/positives').glob('*.fits'))


def load_times_and_fluxes_from_path(path):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def positive_label_function(path):
    return 1


def negative_label_function(path):
    return 0


def get_transit_train_dataset():
    positive_train_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_positive_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_train_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_negative_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_train_light_curve_collection,
                                          negative_train_light_curve_collection])
    return train_light_curve_dataset


def get_transit_validation_dataset():
    positive_validation_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_positive_validation_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_validation_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_negative_validation_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    validation_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_validation_light_curve_collection,
                                          negative_validation_light_curve_collection])
    return validation_light_curve_dataset


def get_transit_finite_test_dataset():
    positive_test_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_positive_test_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_test_light_curve_collection = LabeledLightCurveCollection.new(
        get_paths_function=get_negative_test_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    test_light_curve_dataset = FiniteStandardLightCurveObservationDataset.new(
        standard_light_curve_collections=[positive_test_light_curve_collection,
                                          negative_test_light_curve_collection])
    return test_light_curve_dataset
