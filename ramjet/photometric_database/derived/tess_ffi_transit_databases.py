from ramjet.photometric_database.derived.tess_ffi_eclipsing_binary_lightcurve_collection import \
    TessFfiAntiEclipsingBinaryForTransitLightcurveCollection, TessFfiQuickTransitNegativeLightcurveCollection
from ramjet.photometric_database.derived.tess_ffi_lightcurve_collection import TessFfiLightcurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_lightcurve_collections import \
    TessFfiConfirmedTransitLightcurveCollection, TessFfiNonTransitLightcurveCollection
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
        self.shuffle_buffer_size = 10000
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION


magnitude_range = (0, 14)


class TessFfiStandardTransitDatabase(TessFfiDatabase):
    """
    A database using standard positive and negative transit lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range)
        ]


class TessFfiStandardTransitAntiEclipsingBinaryDatabase(TessFfiDatabase):
    """
    A database using standard positive and negative transit lightcurves and a negative eclipsing binary collection.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=list(range(8)),
                                                                     magnitude_range=magnitude_range)
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=[8],
                                                                     magnitude_range=magnitude_range)
        ]
        self.inference_lightcurve_collections = [
            TessFfiLightcurveCollection(dataset_splits=[9], magnitude_range=magnitude_range)]


class TessFfiInjectedTransitDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_injectee_lightcurve_collection = TessFfiNonTransitLightcurveCollection(
            dataset_splits=list(range(8)))
        self.training_injectable_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range)
        ]


class TessFfiStandardAndInjectedTransitDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        ]
        self.training_injectee_lightcurve_collection = TessFfiNonTransitLightcurveCollection(
            dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        self.training_injectable_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range)
        ]


class TessFfiStandardAndInjectedTransitAntiEclipsingBinaryDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit lightcurves injected into negative lightcurves.
    """
    def __init__(self):
        super().__init__()
        self.shuffle_buffer_size = 10000
        self.number_of_parallel_processes_per_map = 6
        self.training_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=list(range(8)),
                                                                     magnitude_range=magnitude_range),
            # TessFfiQuickTransitNegativeLightcurveCollection(dataset_splits=list(range(8)),
            #                                                 magnitude_range=magnitude_range)
        ]
        self.training_injectee_lightcurve_collection = TessFfiNonTransitLightcurveCollection(
            dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        self.training_injectable_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=list(range(8)),
                                                                     magnitude_range=magnitude_range),
            # TessFfiQuickTransitNegativeLightcurveCollection(dataset_splits=list(range(8)),
            #                                                 magnitude_range=magnitude_range)
        ]
        self.validation_standard_lightcurve_collections = [
            TessFfiConfirmedTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightcurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightcurveCollection(dataset_splits=list(range(8)),
                                                                     magnitude_range=magnitude_range),
            # TessFfiQuickTransitNegativeLightcurveCollection(dataset_splits=list(range(8)),
            #                                                 magnitude_range=magnitude_range)
        ]
        self.inference_lightcurve_collections = [TessFfiLightcurveCollection(magnitude_range=magnitude_range)]
