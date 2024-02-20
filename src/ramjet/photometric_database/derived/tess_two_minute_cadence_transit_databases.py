from ramjet.photometric_database.derived.tess_two_minute_cadence_light_curve_collection import (
    TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection,
)
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_light_curve_collections import (
    TessTwoMinuteCadenceConfirmedTransitLightCurveCollection,
    TessTwoMinuteCadenceNonTransitLightCurveCollection,
)
from ramjet.photometric_database.light_curve_dataset_manipulations import OutOfBoundsInjectionHandlingMethod
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class TessTwoMinuteCadenceStandardTransitDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database using standard positive and negative transit light curves.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=list(range(8))),
        ]
        self.validation_standard_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=[8]),
        ]
        self.inference_light_curve_collections = [
            TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection(dataset_splits=[9])
        ]


class TessTwoMinuteCadenceInjectedTransitDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database using positive and negative transit light curves injected into negative light curves.
    """

    def __init__(self):
        super().__init__()
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        self.training_injectee_light_curve_collection = TessTwoMinuteCadenceNonTransitLightCurveCollection(
            dataset_splits=list(range(8))
        )
        self.training_injectable_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=list(range(8))),
        ]
        self.validation_standard_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=[8]),
        ]
        self.inference_light_curve_collections = [
            TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection(dataset_splits=[9])
        ]


class TessTwoMinuteCadenceStandardAndInjectedTransitDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database using standard positive and negative transit light curves and positives injected into negatives.
    """

    def __init__(self):
        super().__init__()
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        self.training_standard_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=list(range(8))),
        ]
        self.training_injectee_light_curve_collection = TessTwoMinuteCadenceNonTransitLightCurveCollection(
            dataset_splits=list(range(8))
        )
        self.training_injectable_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=list(range(8))),
        ]
        self.validation_standard_light_curve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightCurveCollection(dataset_splits=[8]),
        ]
        self.inference_light_curve_collections = [
            TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection(dataset_splits=[9])
        ]
