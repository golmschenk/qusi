"""
An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections.
"""
import tensorflow as tf
from typing import List, Union

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection
from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class StandardAndInjectedLightcurveDatabase(LightcurveDatabase):
    """
    An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections
    to be used for training.
    """
    def __init__(self):
        super().__init__()
        self.training_standard_lightcurve_collections: List[LightcurveCollection] = []
        self.training_injectee_lightcurve_collection: Union[LightcurveCollection, None] = None
        self.training_injectable_lightcurve_collections: List[LightcurveCollection] = []
        self.validation_standard_lightcurve_collections: List[LightcurveCollection] = []
        self.validation_injectee_lightcurve_collection: Union[LightcurveCollection, None] = None
        self.validation_injectable_lightcurve_collections: List[LightcurveCollection] = []

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        pass

    def generate_paths_datasets_from_lightcurve_collection_group(
                self, standard_lightcurve_collections: List[LightcurveCollection],
                injectee_lightcurve_collection: LightcurveCollection,
                injectable_lightcurve_collections: List[LightcurveCollection]
            ) -> (List[tf.data.Dataset], tf.data.Dataset, List[tf.data.Dataset]):
        standard_paths_datasets = [self.paths_dataset_from_list_or_generator_factory(collection.get_lightcurve_paths)
                                   for collection in standard_lightcurve_collections]
        injectee_path_dataset = self.paths_dataset_from_list_or_generator_factory(
            injectee_lightcurve_collection.get_lightcurve_paths)
        injectable_paths_datasets = [self.paths_dataset_from_list_or_generator_factory(collection.get_lightcurve_paths)
                                   for collection in injectable_lightcurve_collections]
        return standard_paths_datasets, injectee_path_dataset, injectable_paths_datasets
