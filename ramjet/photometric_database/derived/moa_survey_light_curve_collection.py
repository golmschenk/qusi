import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Iterable

from ramjet.data_interface.moa_data_interface import MoaDataInterface
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


class MoaSurveyLightCurveCollection(LightcurveCollection):
    """
    A collection of light curves based on the MOA 9-year survey.
    """
    moa_data_interface = MoaDataInterface()

    def __init__(self, survey_tags: List[str], dataset_splits: Union[List[int], None] = None):
        super().__init__()
        self.survey_tags: List[str] = survey_tags
        self.dataset_splits: Union[List[int], None] = dataset_splits

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        paths: List[Path] = []
        for tag in self.survey_tags:
            tag_paths = self.moa_data_interface.survey_tag_to_path_list_dictionary[tag]
            if self.dataset_splits is not None:
                # Split on each tag, so that the splitting remains across collections with different tag selections.
                tag_paths = self.shuffle_and_split_paths(tag_paths, self.dataset_splits)
            paths.extend(tag_paths)
        return paths

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        lightcurve_dataframe = pd.read_feather(path)
        times = lightcurve_dataframe['HJD'].values
        fluxes = lightcurve_dataframe['flux'].values
        return times, fluxes
