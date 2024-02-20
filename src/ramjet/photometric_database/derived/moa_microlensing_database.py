"""
Code for creating database based on the MOA microlensing data.
"""
from ramjet.photometric_database.derived.moa_microlensing_light_curve_collection import (
    MicrolensingSyntheticGeneratedDuringRunningSignalCollection,
    MOANegativeMicrolensingLightCurveCollection,
    MOAPositiveMicrolensingLightCurveCollection,
)
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class MoaMicrolensingDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database to train a network to find microlensing events in MOA data. Uses real data that were previous classified.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            MOAPositiveMicrolensingLightCurveCollection(dataset_splits=[0, 2, 3, 4], split_pieces=5),
            MOANegativeMicrolensingLightCurveCollection(dataset_splits=[0, 2, 3, 4], split_pieces=5),
        ]
        self.validation_standard_light_curve_collections = [
            MOAPositiveMicrolensingLightCurveCollection(dataset_splits=[1], split_pieces=5),
            MOANegativeMicrolensingLightCurveCollection(dataset_splits=[1], split_pieces=5),
        ]
        self.inference_light_curve_collections = MOANegativeMicrolensingLightCurveCollection(
            dataset_splits=[1], split_pieces=5
        )


class MoaMicrolensingWithSyntheticDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database to train a network to find microlensing events in MOA data. Uses synthetic data injected on previous
    data classified as negative.
    """

    def __init__(self):
        super().__init__()
        self.allow_out_of_bounds_injection = True
        self.training_standard_light_curve_collections = [
            MOAPositiveMicrolensingLightCurveCollection(),
            MOANegativeMicrolensingLightCurveCollection(),
        ]
        self.training_injectee_light_curve_collection = MOANegativeMicrolensingLightCurveCollection()
        self.training_injectable_light_curve_collections = MicrolensingSyntheticGeneratedDuringRunningSignalCollection()
        self.validation_standard_light_curve_collections = self.training_standard_light_curve_collections
