"""
Code for a lightcurve collection of the TESS FFI data, as produced by Brian Powell.
"""
import numpy as np
from pathlib import Path
from typing import Union, List
from peewee import Select

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.data_interface.tess_ffi_lightcurve_metadata_manager import TessFfiLightcurveMetadataManager, \
    TessFfiLightcurveMetadata
from ramjet.photometric_database.sql_metadata_lightcurve_collection import SqlMetadataLightcurveCollection
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve


class TessFfiLightcurveCollection(SqlMetadataLightcurveCollection):
    """
    A lightcurve collection of the TESS two minute cadence data.
    """
    tess_ffi_lightcurve_metadata_manger = TessFfiLightcurveMetadataManager()

    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__()
        self.data_directory: Path = Path('data/tess_ffi_lightcurves')
        self.label = 0
        self.dataset_splits: Union[List[int], None] = dataset_splits
        self.magnitude_range: (Union[float, None], Union[float, None]) = magnitude_range

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = TessFfiLightcurveMetadata().select()
        if self.magnitude_range[0] is not None and self.magnitude_range[1] is not None:
            query = query.where(TessFfiLightcurveMetadata.magnitude.between(*self.magnitude_range))
        elif self.magnitude_range[0] is not None:
            query = query.where(TessFfiLightcurveMetadata.magnitude > self.magnitude_range[0])
        elif self.magnitude_range[1] is not None:
            query = query.where(TessFfiLightcurveMetadata.magnitude < self.magnitude_range[1])
        if self.dataset_splits is not None:
            query = query.where(TessFfiLightcurveMetadata.dataset_split.in_(self.dataset_splits))
        return query

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the lightcurve path from the SQL database model.

        :return: The path to the lightcurve.
        """
        return Path(self.tess_ffi_lightcurve_metadata_manger.
                    lightcurve_root_directory_path.joinpath(model.path))

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = TessFfiLightCurve.load_fluxes_and_times_from_pickle_file(path)
        return times, fluxes

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        fluxes, times = TessFfiLightCurve.load_fluxes_and_times_from_pickle_file(path)
        magnifications, times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return times, magnifications
