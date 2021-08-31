from ramjet.photometric_database.derived.toy_light_curve_collection import ToyFlatLightCurveCollection, \
    ToySineWaveLightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class ToyDatabase(StandardAndInjectedLightCurveDatabase):
    def __init__(self):
        super().__init__()
        self.time_steps_per_example = 100
        self.training_standard_light_curve_collections = [
            ToyFlatLightCurveCollection(),
            ToySineWaveLightCurveCollection(),
        ]
        self.validation_standard_light_curve_collections = [
            ToyFlatLightCurveCollection(),
            ToySineWaveLightCurveCollection(),
        ]
        self.inference_light_curve_collections = [
            ToyFlatLightCurveCollection(),
            ToySineWaveLightCurveCollection(),
        ]
