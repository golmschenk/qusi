from ramjet.photometric_database.derived.tess_ffi_eclipsing_binary_lightcurve_collection import \
    TessFfiAntiEclipsingBinaryForTransitLightcurveCollection
from ramjet.photometric_database.derived.tess_ffi_lightcurve_collection import TessFfiLightcurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_lightcurve_collections import \
    TessFfiConfirmedAndCandidateTransitLightcurveCollection, TessFfiNonTransitLightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase, \
    OutOfBoundsInjectionHandlingMethod


class TessFfiDatabase(StandardAndInjectedLightcurveDatabase):
    """
    An abstract database with settings preset for the FFI data.
    """
    def __init__(self):
        super().__init__()
        self.batch_size = 1000
        self.time_steps_per_example = 1000
        self.shuffle_buffer_size = 100000
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION


class TessFfiStandardTransitDatabase(TessFfiDatabase):
    """
    A database using standard positive and negative transit lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8])
        ]


class TessFfiStandardTransitAntiEclipsingBinaryDatabase(TessFfiDatabase):
    """
    A database using standard positive and negative transit lightcurves and a negative eclipsing binary collection.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedAndCandidateTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8]),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=[8])
        ]
        self.inference_lightcurve_collection = TessFfiLightcurveCollection(dataset_splits=[9])


class TessFfiInjectedTransitDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
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


class TessFfiStandardAndInjectedTransitDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
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
