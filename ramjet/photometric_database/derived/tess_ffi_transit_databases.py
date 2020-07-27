from ramjet.photometric_database.derived.tess_ffi_eclipsing_binary_lightcurve_collection import \
    TessFfiEclipsingBinaryNegativeLabelLightcurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_lightcurve_collections import \
    TessFfiConfirmedAndCandidateTransitLightcurveCollection, TessFfiNonTransitLightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase, \
    OutOfBoundsInjectionHandlingMethod


class TessFfiStandardTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using standard positive and negative transit lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.batch_size = 1000
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8])
        ]


class TessFfiStandardTransitAntiEclipsingBinaryDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using standard positive and negative transit lightcurves and a negative eclipsing binary collection.
    """
    def __init__(self):
        super().__init__()
        self.batch_size = 1000
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiEclipsingBinaryNegativeLabelLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiEclipsingBinaryNegativeLabelLightcurveCollection(dataset_splits=[8])
        ]


class TessFfiInjectedTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.batch_size = 1000
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.
        self.training_injectee_lightcurve_collection = TessFfiNonTransitLightcurveCollection(
            dataset_splits=list(range(8)))
        self.training_injectable_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8])
        ]


class TessFfiStandardAndInjectedTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.batch_size = 1000
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.training_injectee_lightcurve_collection = TessFfiNonTransitLightcurveCollection(
            dataset_splits=list(range(8)))
        self.training_injectable_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8])
        ]
