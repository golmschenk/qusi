from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_lightcurve_collection import \
    TessTwoMinuteCadenceNonTransitLightcurveCollection, TessTwoMinuteCadenceTransitLightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


class TessTwoMinuteCadenceStandardTransitDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database to train a network to find self lensing binaries in TESS two minute cadence data.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            TessTwoMinuteCadenceTransitLightcurveCollection(dataset_splits=list(range(8))),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=list(range(8)))
        ]
        self.validation_standard_lightcurve_collections = [
            TessTwoMinuteCadenceTransitLightcurveCollection(dataset_splits=[8]),
            TessTwoMinuteCadenceNonTransitLightcurveCollection(dataset_splits=[8])
        ]
