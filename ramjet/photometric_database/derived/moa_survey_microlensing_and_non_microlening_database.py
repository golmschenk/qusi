"""
Code for a database of MOA light curves including non-microlensing, and microlensing collections.
"""
from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.derived.moa_survey_light_curve_collection import MoaSurveyLightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class MoaSurveyMicrolensingAndNonMicrolensingDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A class for a database of MOA light curves including non-microlensing, and microlensing collections.
    """
    moa_data_interface = MoaDataInterface()

    def __init__(self, test_split: int):
        super().__init__()
        validation_split = (test_split - 1) % 10
        train_splits = list(range(10))
        train_splits.remove(validation_split)
        train_splits.remove(test_split)
        self.number_of_label_values = 1
        self.number_of_parallel_processes_per_map = 5
        self.time_steps_per_example = 18000
        self.shuffle_buffer_size = 1000
        self.include_time_as_channel = False

        # Note that the NN has number_of_splits: int = 10 already set.
        # Creating the training collection | splits [0, 1, 2, 3, 4, 5, 6, 7] = 80% of the data
        negative_training = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=train_splits)
        positive_training = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=train_splits)
        self.training_standard_light_curve_collections = [negative_training, positive_training]

        # Creating the validation collection | split [8] = 10% of the data
        negative_validation = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=[validation_split])
        positive_validation = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[validation_split])
        self.validation_standard_light_curve_collections = [negative_validation, positive_validation]

        # Creating the inference collection | split [9] = 10% of the data
        negative_inference = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=[test_split])
        positive_inference = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[test_split])
        self.inference_light_curve_collections = [negative_inference, positive_inference]


class MoaSurveyMicrolensingAndNonMicrolensingWithHardCasesDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A class for a database of MOA light curves including non-microlensing, and microlensing collections.
    """
    moa_data_interface = MoaDataInterface()

    def __init__(self, test_split: int):
        super().__init__()
        validation_split = (test_split - 1) % 10
        train_splits = list(range(10))
        train_splits.remove(validation_split)
        train_splits.remove(test_split)
        self.number_of_label_values = 1
        self.number_of_parallel_processes_per_map = 5
        self.time_steps_per_example = 18000
        self.shuffle_buffer_size = 1000
        self.include_time_as_channel = False

        # Note that the NN has number_of_splits: int = 10 already set.
        # Creating the training collection | splits [0, 1, 2, 3, 4, 5, 6, 7] = 80% of the data
        negative_training = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=train_splits)
        negative_hardcases_training = MoaSurveyLightCurveCollection(
            survey_tags=['n', 'nr'],
            label=0,
            dataset_splits=train_splits)
        positive_training = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=train_splits)
        self.training_standard_light_curve_collections = [negative_training,
                                                          negative_hardcases_training,
                                                          positive_training]

        # Creating the validation collection | split [8] = 10% of the data
        negative_validation = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=[validation_split])
        negative_hardcases_validation = MoaSurveyLightCurveCollection(
            survey_tags=['n', 'nr'],
            label=0,
            dataset_splits=[validation_split])
        positive_validation = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[validation_split])
        self.validation_standard_light_curve_collections = [negative_validation,
                                                            negative_hardcases_validation,
                                                            positive_validation]

        # Creating the inference collection | split [9] = 10% of the data
        negative_inference = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=[test_split])
        negative_hardcases_inference = MoaSurveyLightCurveCollection(
            survey_tags=['n', 'nr'],
            label=0,
            dataset_splits=[test_split])
        positive_inference = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[test_split])
        self.inference_light_curve_collections = [negative_inference,
                                                  negative_hardcases_inference,
                                                  positive_inference]


