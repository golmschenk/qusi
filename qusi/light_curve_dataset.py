from typing import List

from torch.utils.data import Dataset, Subset, IterableDataset

from ramjet.photometric_database.light_curve_collection import LightCurveCollection


class LightCurveDataset(IterableDataset):
    def __init__(self):
        self.standard_light_curve_collections: List[LightCurveCollection] = []
        self.injectee_light_curve_collection: List[LightCurveCollection] = []
        self.injectable_light_curve_collections: List[LightCurveCollection] = []
        self.include_standard_in_injectee = False  # TODO: Should this be automatically detected?


