from __future__ import annotations

from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.derived.moa_survey_light_curve_collection import MoaSurveyLightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class MoaSurveyBalancedTagDatabase(StandardAndInjectedLightCurveDatabase):
    """
    A database to train a network to find MOA events, balancing the training of each tag.
    """

    moa_data_interface = MoaDataInterface()

    def __init__(self):
        super().__init__()
        self.number_of_label_values = 2
        self.number_of_parallel_processes_per_map = 3
        self.time_steps_per_example = 18000
        self.training_standard_light_curve_collections = self.create_collection_for_each_tag(
            dataset_splits=list(range(8))
        )
        self.validation_standard_light_curve_collections = self.create_collection_for_each_tag(dataset_splits=[8])
        self.inference_light_curve_collections = self.create_collection_for_each_tag(dataset_splits=[9])

    def create_collection_for_each_tag(self, dataset_splits: list[int] | None) -> list[MoaSurveyLightCurveCollection]:
        """
        Creates a light curve collection for each tag in the survey and assigns the appropriate labels.

        :param dataset_splits: The dataset splits to include in each collection.
        :return: The list of collections.
        """
        collections = []
        for tag in self.moa_data_interface.survey_tag_to_path_list_dictionary:
            if tag == "i":
                continue  # There is a single `i` tag, which seems to be a typo.
            collection = MoaSurveyLightCurveCollection(survey_tags=[tag], dataset_splits=dataset_splits)
            if tag == "cb":
                collection.label = [1, 1]
            elif tag in ["c", "cf", "cp", "cw", "cs"]:
                collection.label = [1, 0]
            else:
                collection.label = [0, 0]
            collections.append(collection)
        return collections
