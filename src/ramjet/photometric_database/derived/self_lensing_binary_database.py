from ramjet.data_interface.tess_data_interface import TessFluxType
from ramjet.photometric_database.derived.self_lensing_binary_synthetic_signals_light_curve_collection import (
    ReversedSelfLensingBinarySyntheticSignalsLightCurveCollection,
    SelfLensingBinarySyntheticSignalsLightCurveCollection,
)
from ramjet.photometric_database.derived.tess_two_minute_cadence_light_curve_collection import (
    TessTwoMinuteCadenceLightCurveCollection,
)
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class SelfLensingBinaryDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database to train a network to find self lensing binaries in TESS two minute cadence data.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessTwoMinuteCadenceLightCurveCollection(flux_type=TessFluxType.SAP)
        ]
        self.training_injectee_light_curve_collection = TessTwoMinuteCadenceLightCurveCollection(
            flux_type=TessFluxType.SAP
        )
        self.training_injectable_light_curve_collections = [
            SelfLensingBinarySyntheticSignalsLightCurveCollection(),
            ReversedSelfLensingBinarySyntheticSignalsLightCurveCollection(),
        ]
        self.validation_standard_light_curve_collections = self.training_standard_light_curve_collections
        self.validation_injectee_light_curve_collection = self.training_injectee_light_curve_collection
        self.validation_injectable_light_curve_collections = self.training_injectable_light_curve_collections
        self.inference_light_curve_collections = [TessTwoMinuteCadenceLightCurveCollection(flux_type=TessFluxType.SAP)]
