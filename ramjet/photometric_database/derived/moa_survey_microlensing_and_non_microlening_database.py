"""
Code for a database of MOA light curves including non-microlensing, and microlensing collections.
"""
from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.derived.moa_survey_light_curve_collection import MoaSurveyLightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class MoaSurveyMicrolensingAndNonMicroleningDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A class for a database of MOA light curves including non-microlensing, and microlensing collections.
    """
    moa_data_interface = MoaDataInterface()

    def __init__(self):
        super().__init__()
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
            dataset_splits=list(range(8)))
        positive_training = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=list(range(8)))
        self.training_standard_light_curve_collections = [negative_training, positive_training]

        # Creating the validation collection | split [8] = 10% of the data
        negative_validation = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=[8])
        positive_validation = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[8])
        self.validation_standard_light_curve_collections = [negative_validation, positive_validation]

        # Creating the inference collection | split [9] = 10% of the data
        negative_inference = MoaSurveyLightCurveCollection(
            survey_tags=['v', 'n', 'nr', 'm', 'j', self.moa_data_interface.no_tag_string],
            label=0,
            dataset_splits=[9])
        positive_inference = MoaSurveyLightCurveCollection(
            survey_tags=['c', 'cf', 'cp', 'cw', 'cs', 'cb'],
            label=1,
            dataset_splits=[9])
        self.inference_light_curve_collections = [negative_inference, positive_inference]
