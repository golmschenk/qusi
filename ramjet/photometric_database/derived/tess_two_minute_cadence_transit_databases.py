from ramjet.photometric_database.derived.tess_two_minute_cadence_lightcurve_collection import \
    TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_lightcurve_collections import \
    TessTwoMinuteCadenceNonTransitLightcurveCollection, \
    TessTwoMinuteCadenceConfirmedTransitLightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import \
    StandardAndInjectedLightcurveDatabase, OutOfBoundsInjectionHandlingMethod


class TessTwoMinuteCadenceStandardTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using standard positive and negative transit lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=[8])
        ]
        self.inference_lightcurve_collection = TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection(
            dataset_splits=[9])


class TessTwoMinuteCadenceInjectedTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        self.training_injectee_lightcurve_collection = TessTwoMinuteCadenceNonTransitLightcurveCollection(
            dataset_splits=list(range(8)))
        self.training_injectable_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=[8])
        ]
        self.inference_lightcurve_collection = TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection(
            dataset_splits=[9])


class TessTwoMinuteCadenceStandardAndInjectedTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using standard positive and negative transit lightcurves and positives injected into negatives.
    """
    def __init__(self):
        super().__init__()
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        self.training_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.training_injectee_lightcurve_collection = TessTwoMinuteCadenceNonTransitLightcurveCollection(
            dataset_splits=list(range(8)))
        self.training_injectable_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedTransitLightcurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=[8])
        ]
        self.inference_lightcurve_collection = TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection(
            dataset_splits=[9])
