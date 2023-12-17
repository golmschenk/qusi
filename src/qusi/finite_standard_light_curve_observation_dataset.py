from dataclasses import dataclass
from typing import List, Callable, Any, Self

import numpy as np
from torch.utils.data import Dataset

from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_dataset import default_light_curve_post_injection_transform


@dataclass
class FiniteStandardLightCurveObservationDataset(Dataset):
    standard_light_curve_collections: List[LabeledLightCurveCollection]
    post_injection_transform: Callable[[Any], Any]
    length: int
    collection_start_indexes: List[int]

    @classmethod
    def new(cls, standard_light_curve_collections: List[LabeledLightCurveCollection]) -> Self:
        length = 0
        collection_start_indexes: List[int] = []
        for standard_light_curve_collection in standard_light_curve_collections:
            standard_light_curve_collection_length = len(
                list(standard_light_curve_collection.path_getter.get_paths()))
            collection_start_indexes.append(length)
            length += standard_light_curve_collection_length
        instance = cls(standard_light_curve_collections=standard_light_curve_collections,
                       post_injection_transform=default_light_curve_post_injection_transform,
                       length=length,
                       collection_start_indexes=collection_start_indexes)
        return instance

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Any:
        collection_index = np.searchsorted(self.collection_start_indexes, index, side='right') - 1
        collection = self.standard_light_curve_collections[collection_index]
        index_in_collection = index - self.collection_start_indexes[collection_index]
        observation = collection[index_in_collection]
        transformed_observation = self.post_injection_transform(observation)
        return transformed_observation