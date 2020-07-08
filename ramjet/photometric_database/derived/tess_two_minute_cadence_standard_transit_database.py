from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_lightcurve_collections import \
    TessTwoMinuteCadenceNonTransitLightcurveCollection, TessTwoMinuteCadenceConfirmedTransitLightcurveCollection, \
    TessTwoMinuteCadenceConfirmedAndCandidateTransitLightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


class TessTwoMinuteCadenceStandardTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using standard positive and negative transit lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=[8])
        ]


class TessTwoMinuteCadenceInjectedTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_injectee_lightcurve_collection = TessTwoMinuteCadenceNonTransitLightcurveCollection(
            dataset_splits=list(range(8)))
        self.training_injectable_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessTwoMinuteCadenceConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=[8])
        ]
