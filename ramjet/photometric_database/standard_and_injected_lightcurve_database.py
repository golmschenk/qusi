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
        (training_standard_paths_datasets, training_injectee_path_dataset,
         training_injectable_paths_datasets) = self.generate_paths_datasets_from_lightcurve_collection_group(
            self.training_standard_lightcurve_collections,
            self.training_injectee_lightcurve_collection,
            self.training_injectable_lightcurve_collections)
        (validation_standard_paths_datasets, validation_injectee_path_dataset,
         validation_injectable_paths_datasets) = self.generate_paths_datasets_from_lightcurve_collection_group(
            self.validation_standard_lightcurve_collections,
            self.validation_injectee_lightcurve_collection,
            self.validation_injectable_lightcurve_collections)

    def generate_paths_datasets_from_lightcurve_collection_group(
            self, standard_lightcurve_collections: List[LightcurveCollection],
            injectee_lightcurve_collection: LightcurveCollection,
            injectable_lightcurve_collections: List[LightcurveCollection]
    ) -> (List[tf.data.Dataset], tf.data.Dataset, List[tf.data.Dataset]):
        """
        Produces the paths datasets for a group of lightcurve collections.

        :param standard_lightcurve_collections: The standard style lightcurve collections.
        :param injectee_lightcurve_collection: The injectee lightcurve collection.
        :param injectable_lightcurve_collections: The injectable lightcurve collections.
        :return: A tuple of the list of standard paths datasets, the injectee paths dataset, and the list of the
                 injectable paths datasets.
        """
        standard_paths_datasets = [self.paths_dataset_from_list_or_generator_factory(collection.get_lightcurve_paths)
                                   for collection in standard_lightcurve_collections]
        injectee_path_dataset = self.paths_dataset_from_list_or_generator_factory(
            injectee_lightcurve_collection.get_lightcurve_paths)
        injectable_paths_datasets = [self.paths_dataset_from_list_or_generator_factory(collection.get_lightcurve_paths)
                                     for collection in injectable_lightcurve_collections]
        return standard_paths_datasets, injectee_path_dataset, injectable_paths_datasets

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
