from ramjet.photometric_database.derived.self_lensing_binary_synthetic_signals_lightcurve_collection import \
    SelfLensingBinarySyntheticSignalsLightcurveCollection, ReversedSelfLensingBinarySyntheticSignalsLightcurveCollection
from ramjet.photometric_database.derived.tess_two_minute_cadence_lightcurve_collection import TessTwoMinuteCadenceLightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


class SelfLensingBinaryDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database to train a network to find self lensing binaries in TESS two minute cadence data.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [TessTwoMinuteCadenceLightcurveCollection()]
        self.training_injectee_lightcurve_collection = TessTwoMinuteCadenceLightcurveCollection()
        self.training_injectable_lightcurve_collections = [
            SelfLensingBinarySyntheticSignalsLightcurveCollection(),
            ReversedSelfLensingBinarySyntheticSignalsLightcurveCollection()
        ]
        self.validation_standard_lightcurve_collections = self.training_standard_lightcurve_collections
        self.validation_injectee_lightcurve_collection = self.training_injectee_lightcurve_collection
        self.validation_injectable_lightcurve_collections = self.training_injectable_lightcurve_collections
