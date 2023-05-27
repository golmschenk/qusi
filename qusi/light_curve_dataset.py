from typing import List

from torch.utils.data import Dataset, Subset, IterableDataset

from ramjet.photometric_database.light_curve_collection import LightCurveCollection


class LightCurveDataset(IterableDataset):
    def __init__(self,
                 standard_light_curve_collections: List[LightCurveCollection],
                 injectee_light_curve_collections: List[LightCurveCollection],
                 injectable_light_curve_collections: List[LightCurveCollection]
                 ):
        self.standard_light_curve_collections: List[LightCurveCollection] = standard_light_curve_collections
        self.injectee_light_curve_collections: List[LightCurveCollection] = injectee_light_curve_collections
        self.injectable_light_curve_collections: List[LightCurveCollection] = injectable_light_curve_collections
        if len(self.standard_light_curve_collections) == 0 and len(self.injectee_light_curve_collections) == 0:
            raise ValueError('Either the standard or injectee light curve collection lists must not be empty. '
                             'Both were empty.')
        self.include_standard_in_injectee = False  # TODO: Should this be automatically detected?

    @classmethod
    def new(cls,
            standard_light_curve_collections: List[LightCurveCollection] | None = None,
            injectee_light_curve_collections: List[LightCurveCollection] | None = None,
            injectable_light_curve_collections: List[LightCurveCollection] | None = None,
            ):
        if standard_light_curve_collections is None and injectee_light_curve_collections is None:
            raise ValueError('Either the standard or injectee light curve collection lists must be specified. '
                             'Both were `None`.')
        if standard_light_curve_collections is None:
            standard_light_curve_collections = []
        if injectee_light_curve_collections is None:
            injectee_light_curve_collections = []
        if injectable_light_curve_collections is None:
            injectable_light_curve_collections = []
        instance = cls(standard_light_curve_collections=standard_light_curve_collections,
                       injectee_light_curve_collections=injectee_light_curve_collections,
                       injectable_light_curve_collections=injectable_light_curve_collections)
        return instance


def is_injected_dataset(dataset: LightCurveDataset):
    return len(dataset.injectee_light_curve_collections) > 0


def contains_injected_dataset(datasets: List[LightCurveDataset]):
    for dataset in datasets:
        if is_injected_dataset(dataset):
            return True
    return False