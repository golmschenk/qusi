from functools import partial

import numpy as np
import pandas as pd

from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase

from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_dataset import LightCurveDataset, default_light_curve_observation_post_injection_transform
from qusi.light_curve_collection import LightCurveCollection



def positive_label_function(path):
    return 1


def negative_label_function(path):
    return 0

class MoaSurveyMicrolensingAndNonMicrolensingDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A class for a database of MOA light curves including non-microlensing, and microlensing collections.
    """

    def __init__(self, test_split: int):
        super().__init__()
        validation_split = (test_split - 1) % 10
        train_splits = list(range(10))
        train_splits.remove(validation_split)
        train_splits.remove(test_split)

        self.negative_training = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', 'no_tag'],
            label=0,
            dataset_splits=train_splits)
        self.positive_training = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=train_splits)

        self.negative_validation = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', 'no_tag'],
            label=0,
            dataset_splits=[validation_split])
        self.positive_validation = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[validation_split])

        self.negative_inference = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', 'no_tag'],
            label=0,
            dataset_splits=[test_split])
        self.positive_inference = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[test_split])
        self.all_inference = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb',
                         'v', 'n', 'nr', 'm', 'j', 'no_tag'],
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
        # print('check "properties" of the train_light_curve_dataset', train_light_curve_dataset)
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

class MoaSurveyLightCurveCollection(LightCurveCollection):
    """
    A collection of light curves based on the MOA 9-year survey.
    """

    def __init__(
        self,
        survey_tags: list[str],
        dataset_splits: list[int] | None = None,
        label: float | list[float] | np.ndarray | None = None,
    ):
        super().__init__()
        self.label = label
        self.survey_tags: list[str] = survey_tags
        self.dataset_splits: list[int] | None = dataset_splits

    def get_paths(self):
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        paths: list[Path] = []
        for tag in self.survey_tags:
            tag_paths = self.moa_data_interface.survey_tag_to_path_list_dictionary[tag]
            if self.dataset_splits is not None:
                # Split on each tag, so that the splitting remains across collections with different tag selections.
                tag_paths = self.shuffle_and_split_paths(tag_paths, self.dataset_splits)
            paths.extend(tag_paths)
        return paths

    def load_times_and_fluxes_from_path(self, path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        light_curve_dataframe = pd.read_feather(path)
        times = light_curve_dataframe["HJD"].values
        fluxes = light_curve_dataframe["flux"].values
        return times, fluxes

