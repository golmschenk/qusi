from ramjet.photometric_database.derived.tess_ffi_eclipsing_binary_light_curve_collection import (
    TessFfiAntiEclipsingBinaryForTransitLightCurveCollection,
)
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_light_curve_collections import (
    TessFfiConfirmedTransitLightCurveCollection,
    TessFfiNonTransitLightCurveCollection,
)
from ramjet.photometric_database.light_curve_dataset_manipulations import OutOfBoundsInjectionHandlingMethod
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class TessFfiDatabase(StandardAndInjectedLightCurveDatabase):
    """
    An abstract database with settings preset for the FFI data.
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 1000
        self.time_steps_per_example = 1000
        self.shuffle_buffer_size = 100000
        self.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION


magnitude_range = (0, 11)


class TessFfiStandardTransitDatabase(TessFfiDatabase):
    """
    A database using standard positive and negative transit light curves.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
        ]


class TessFfiStandardTransitAntiEclipsingBinaryDatabase(TessFfiDatabase):
    """
    A database using standard positive and negative transit light curves and a negative eclipsing binary collection.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightCurveCollection(
                dataset_splits=list(range(8)), magnitude_range=magnitude_range
            ),
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightCurveCollection(
                dataset_splits=[8], magnitude_range=magnitude_range
            ),
        ]
        self.inference_light_curve_collections = [
            TessFfiLightCurveCollection(dataset_splits=[9], magnitude_range=magnitude_range)
        ]


class TessFfiInjectedTransitDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit light curves injected into negative light curves.
    """

    def __init__(self):
        super().__init__()
        self.training_injectee_light_curve_collection = TessFfiNonTransitLightCurveCollection(
            dataset_splits=list(range(8))
        )
        self.training_injectable_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
        ]


class TessFfiStandardAndInjectedTransitDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit light curves injected into negative light curves.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
        ]
        self.training_injectee_light_curve_collection = TessFfiNonTransitLightCurveCollection(
            dataset_splits=list(range(8)), magnitude_range=magnitude_range
        )
        self.training_injectable_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
        ]


class TessFfiStandardAndInjectedTransitAntiEclipsingBinaryDatabase(TessFfiDatabase):
    """
    A database using positive and negative transit light curves injected into negative light curves.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightCurveCollection(
                dataset_splits=list(range(8)), magnitude_range=magnitude_range
            ),
        ]
        self.training_injectee_light_curve_collection = TessFfiNonTransitLightCurveCollection(
            dataset_splits=list(range(8)), magnitude_range=magnitude_range
        )
        self.training_injectable_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightCurveCollection(
                dataset_splits=list(range(8)), magnitude_range=magnitude_range
            ),
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiConfirmedTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiNonTransitLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiAntiEclipsingBinaryForTransitLightCurveCollection(
                dataset_splits=list(range(8)), magnitude_range=magnitude_range
            ),
        ]
