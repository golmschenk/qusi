"""
Code for creating database based on the MOA microlensing data.
"""
from ramjet.photometric_database.derived.moa_microlensing_lightcurve_collection import \
    MOAPositiveMicrolensingLightcurveCollection, MOANegativeMicrolensingLightcurveCollection, \
    MicrolensingSyntheticGeneratedDuringRunningSignalCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


class MoaMicrolensingDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database to train a network to find microlensing events in MOA data. Uses real data that were previous classified.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections = [
            MOAPositiveMicrolensingLightcurveCollection(),
            MOANegativeMicrolensingLightcurveCollection()
        ]
        self.validation_standard_lightcurve_collections = self.training_standard_lightcurve_collections


class MoaMicrolensingWithSyntheticDatabase(StandardAndInjectedLightcurveDatabase):
    """
    A database to train a network to find microlensing events in MOA data. Uses synthetic data injected on previous
    data classified as negative.
    """
    def __init__(self):
        super().__init__()
        self.allow_out_of_bounds_injection = True
        self.training_standard_lightcurve_collections = [
            MOAPositiveMicrolensingLightcurveCollection(),
            MOANegativeMicrolensingLightcurveCollection()
        ]
        self.training_injectee_lightcurve_collection = MOANegativeMicrolensingLightcurveCollection()
        self.training_injectable_lightcurve_collections = [MicrolensingSyntheticGeneratedDuringRunningSignalCollection()]
        self.validation_standard_lightcurve_collections = self.training_standard_lightcurve_collections
