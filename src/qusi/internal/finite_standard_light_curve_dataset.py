from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Self

from qusi.internal.light_curve_collection import LightCurveCollection
from qusi.internal.light_curve_dataset import default_light_curve_post_injection_transform


@dataclass
class FiniteStandardLightCurveDataset(Dataset):
    """
    A finite light curve dataset without injection.
    """
    standard_light_curve_collections: list[LightCurveCollection]
    post_injection_transform: Callable[[Any], Any]
    length: int
    collection_start_indexes: list[int]

    @classmethod
    def new(
            cls,
            light_curve_collections: list[LightCurveCollection],
            *,
            post_injection_transform: Callable[[Any], Any] | None = None,
    ) -> Self:
        """
        Creates a new `FiniteStandardLightCurveDataset`.

        :param light_curve_collections: The light curve collections to include in the dataset.
        :param post_injection_transform: Transforms to the data to occur after injection.
        :return: The dataset.
        """
        if post_injection_transform is None:
            post_injection_transform = partial(default_light_curve_post_injection_transform, length=3500,
                                               randomize=False)
        length = 0
        collection_start_indexes: list[int] = []
        for light_curve_collection in light_curve_collections:
            standard_light_curve_collection_length = len(list(light_curve_collection.path_getter.get_paths()))
            collection_start_indexes.append(length)
            length += standard_light_curve_collection_length
        instance = cls(
            standard_light_curve_collections=light_curve_collections,
            post_injection_transform=post_injection_transform,
            length=length,
            collection_start_indexes=collection_start_indexes,
        )
        return instance

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Any:
        collection_index = np.searchsorted(self.collection_start_indexes, index, side="right") - 1
        collection = self.standard_light_curve_collections[collection_index]
        index_in_collection = index - self.collection_start_indexes[collection_index]
        observation = collection[index_in_collection]
        transformed_observation = self.post_injection_transform(observation)
        return transformed_observation
