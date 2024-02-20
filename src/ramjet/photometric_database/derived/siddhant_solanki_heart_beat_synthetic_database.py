from ramjet.photometric_database.derived.siddhant_solanki_heart_beat_synthetic_signals_collection import (
    SiddhantSolankiHeartBeatSyntheticSignalsCollection,
    SiddhantSolankiNonHeartBeatSyntheticSignalsCollection,
    TessFfiHeartBeatHardNegativeLightcurveCollection,
)
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_databases import TessFfiDatabase

magnitude_range = (0, 15)


class SiddhantSolankiHeartBeatSyntheticDatabase(TessFfiDatabase):
    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TessFfiLightCurveCollection(magnitude_range=magnitude_range),
            TessFfiHeartBeatHardNegativeLightcurveCollection(magnitude_range=magnitude_range),
        ]
        self.training_injectee_light_curve_collection = TessFfiLightCurveCollection(magnitude_range=magnitude_range)
        self.training_injectable_light_curve_collections = [
            SiddhantSolankiHeartBeatSyntheticSignalsCollection(),
            SiddhantSolankiNonHeartBeatSyntheticSignalsCollection(),
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiLightCurveCollection(magnitude_range=magnitude_range)
        ]
        self.validation_injectee_light_curve_collection = self.training_injectee_light_curve_collection
        self.validation_injectable_light_curve_collections = self.training_injectable_light_curve_collections
        self.inference_light_curve_collections = [TessFfiLightCurveCollection(magnitude_range=magnitude_range)]
