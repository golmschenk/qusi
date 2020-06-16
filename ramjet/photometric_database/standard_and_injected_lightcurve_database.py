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
        self.shuffle_buffer_size = 10000

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        training_standard_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.training_standard_lightcurve_collections)
        training_injectee_path_dataset = self.generate_paths_dataset_from_lightcurve_collection(
            self.training_injectee_lightcurve_collection)
        training_injectable_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.training_injectable_lightcurve_collections)
        validation_standard_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.validation_standard_lightcurve_collections)
        validation_injectee_path_dataset = self.generate_paths_dataset_from_lightcurve_collection(
            self.validation_injectee_lightcurve_collection)
        validation_injectable_paths_datasets = self.generate_paths_datasets_from_lightcurve_collection_list(
            self.validation_injectable_lightcurve_collections)

    def generate_paths_dataset_from_lightcurve_collection(self, lightcurve_collection: LightcurveCollection
                                                          ) -> tf.data.Dataset:
        """
        Generates a paths dataset for a lightcurve collection.

        :param lightcurve_collection: The lightcurve collection to generate a paths dataset for.
        :return: The paths dataset.
        """
        paths_dataset = self.paths_dataset_from_list_or_generator_factory(lightcurve_collection.get_lightcurve_paths)
        repeated_paths_dataset = paths_dataset.repeat()
        shuffled_paths_dataset = repeated_paths_dataset.shuffle(self.shuffle_buffer_size)
        return shuffled_paths_dataset

    def generate_paths_datasets_from_lightcurve_collection_list(self, lightcurve_collections: List[LightcurveCollection]
                                                                ) -> List[tf.data.Dataset]:
        """
        Generates a paths dataset for each lightcurve collection in a list.

        :param lightcurve_collections: The list of lightcurve collections.
        :return: The list of paths datasets.
        """
        return [self.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
                for lightcurve_collection in lightcurve_collections]
