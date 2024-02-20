"""
Code for a database of MOA light curves including non-microlensing, single lensing, and binary lensing collcetions.
"""
from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.derived.moa_survey_light_curve_collection import MoaSurveyLightCurveCollection
from ramjet.photometric_database.light_curve_dataset_manipulations import (
    BaselineFluxEstimationMethod,
    OutOfBoundsInjectionHandlingMethod,
)
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class MoaSurveyNoneSingleAndBinaryDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A class for a database of MOA light curves including non-microlensing, single lensing, and binary lensing
    collections.
    """

    moa_data_interface = MoaDataInterface()

    def __init__(self):
        super().__init__()
        self.number_of_label_values = 2
        self.number_of_parallel_processes_per_map = 5
        self.time_steps_per_example = 18000
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        self.baseline_flux_estimation_method = BaselineFluxEstimationMethod.MEDIAN_ABSOLUTE_DEVIATION
        self.shuffle_buffer_size = 1000
        # self.include_time_as_channel = True
        # self.include_flux_errors_as_channel = True
        negative_training = MoaSurveyLightCurveCollection(
            survey_tags=["v", "n", "nr", "m", "j", self.moa_data_interface.no_tag_string],
            label=[0, 0],
            dataset_splits=list(range(8)),
        )
        self.training_standard_light_curve_collections = [
            negative_training,
            MoaSurveyLightCurveCollection(
                survey_tags=["c", "cf", "cp", "cw", "cs"], label=[1, 0], dataset_splits=list(range(8))
            ),
            MoaSurveyLightCurveCollection(survey_tags=["cb"], label=[1, 1], dataset_splits=list(range(8))),
        ]
        self.validation_standard_light_curve_collections = [
            MoaSurveyLightCurveCollection(
                survey_tags=["v", "n", "nr", "m", "j", self.moa_data_interface.no_tag_string],
                label=[0, 0],
                dataset_splits=[8],
            ),
            MoaSurveyLightCurveCollection(survey_tags=["c", "cf", "cp", "cw", "cs"], label=[1, 0], dataset_splits=[8]),
            MoaSurveyLightCurveCollection(survey_tags=["cb"], label=[1, 1], dataset_splits=[8]),
        ]
        self.inference_light_curve_collections = [
            MoaSurveyLightCurveCollection(
                survey_tags=["v", "n", "nr", "m", "j", self.moa_data_interface.no_tag_string],
                label=[0, 0],
                dataset_splits=[9],
            ),
            MoaSurveyLightCurveCollection(survey_tags=["c", "cf", "cp", "cw", "cs"], label=[1, 0], dataset_splits=[9]),
            MoaSurveyLightCurveCollection(survey_tags=["cb"], label=[1, 1], dataset_splits=[9]),
        ]
