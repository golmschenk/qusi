from functools import partial

import numpy as np

from moa_data_interface_2M import MoaDataInterface2M
from moa_survey_light_curve_collection_2M import MoaSurveyLightCurveCollection2M
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase

from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_dataset import LightCurveDataset, default_light_curve_observation_post_injection_transform
from qusi.light_curve_collection import LightCurveCollection


def positive_label_function(path):
    return 1


def negative_label_function(path):
    return 0

class MoaSurveyMicrolensingAndNonMicrolensingDatabase2M(StandardAndInjectedLightCurveDatabase):
    """
    A class for a database of MOA light curves including non-microlensing, and microlensing collections.
    """
    moa_data_interface_2M = MoaDataInterface2M()

    def __init__(self, test_split: int):
        super().__init__()
        validation_split = (test_split - 1) % 10
        train_splits = list(range(10))
        train_splits.remove(validation_split)
        train_splits.remove(test_split)

        # Note that the NN has number_of_splits: int = 10 already set.
        # Creating the training collection | splits [0, 1, 2, 3, 4, 5, 6, 7] = 80% of the data
        self.negative_training = MoaSurveyLightCurveCollection2M(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface_2M.no_tag_string],
            label=0,
            dataset_splits=train_splits)
        self.positive_training = MoaSurveyLightCurveCollection2M(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=train_splits)

        # Creating the validation collection | split [8] = 10% of the data
        self.negative_validation = MoaSurveyLightCurveCollection2M(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface_2M.no_tag_string],
            label=0,
            dataset_splits=[validation_split])
        self.positive_validation = MoaSurveyLightCurveCollection2M(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[validation_split])

        # Creating the inference collection | split [9] = 10% of the data
        self.negative_inference = MoaSurveyLightCurveCollection2M(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface_2M.no_tag_string],
            label=0,
            dataset_splits=[test_split])
        self.positive_inference = MoaSurveyLightCurveCollection2M(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[test_split])
        self.all_inference = MoaSurveyLightCurveCollection2M(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb',
                         'v', 'n', 'nr', 'm', 'j', self.moa_data_interface_2M.no_tag_string],
            label=np.nan,
            dataset_splits=[test_split])

    # QUSI structure
    def get_microlensing_train_dataset(self):
        positive_train_light_curve_collection = LabeledLightCurveCollection.new(
            get_paths_function=self.positive_training.get_paths,
            load_times_and_fluxes_from_path_function=self.positive_training.load_times_and_fluxes_from_path,
            load_label_from_path_function=positive_label_function)
        negative_train_light_curve_collection = LabeledLightCurveCollection.new(
            get_paths_function=self.negative_training.get_paths,
            load_times_and_fluxes_from_path_function=self.negative_training.load_times_and_fluxes_from_path,
            load_label_from_path_function=negative_label_function)
        train_light_curve_dataset = LightCurveDataset.new(
            standard_light_curve_collections=[positive_train_light_curve_collection,
                                              negative_train_light_curve_collection],
            post_injection_transform=partial(
                default_light_curve_observation_post_injection_transform, length=18_000))
        return train_light_curve_dataset

    def get_microlensing_validation_dataset(self):
        positive_validation_light_curve_collection = LabeledLightCurveCollection.new(
            get_paths_function=self.positive_validation.get_paths,
            load_times_and_fluxes_from_path_function=self.positive_validation.load_times_and_fluxes_from_path,
            load_label_from_path_function=positive_label_function)
        negative_validation_light_curve_collection = LabeledLightCurveCollection.new(
            get_paths_function=self.negative_validation.get_paths,
            load_times_and_fluxes_from_path_function=self.negative_validation.load_times_and_fluxes_from_path,
            load_label_from_path_function=negative_label_function)
        validation_light_curve_dataset = LightCurveDataset.new(
            standard_light_curve_collections=[positive_validation_light_curve_collection,
                                              negative_validation_light_curve_collection],
            post_injection_transform=partial(
                default_light_curve_observation_post_injection_transform, length=18_000))
        return validation_light_curve_dataset

    def get_microlensing_infer_collection(self):
        infer_light_curve_collection = LightCurveCollection.new(
            get_paths_function=self.all_inference.get_paths,
            load_times_and_fluxes_from_path_function=self.all_inference.load_times_and_fluxes_from_path)
        return infer_light_curve_collection
