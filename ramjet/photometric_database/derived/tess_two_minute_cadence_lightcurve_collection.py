"""
Code for a lightcurve collection of the TESS two minute cadence data.
"""
import numpy as np
from pathlib import Path
from typing import Iterable, Union, List

from peewee import Select

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.data_interface.tess_data_interface import TessDataInterface, TessFluxType
from ramjet.data_interface.tess_target_metadata_manager import TessTargetMetadata
from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import \
    TessTwoMinuteCadenceLightcurveMetadataManger, TessTwoMinuteCadenceLightcurveMetadata
from ramjet.photometric_database.sql_metadata_lightcurve_collection import SqlMetadataLightcurveCollection


class TessTwoMinuteCadenceLightcurveCollection(SqlMetadataLightcurveCollection):
    """
    A lightcurve collection of the TESS two minute cadence data.
    """
    tess_data_interface = TessDataInterface()
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()

    def __init__(self, dataset_splits: Union[List[int], None] = None):
        super().__init__()
        self.data_directory: Path = Path('data/tess_two_minute_cadence_lightcurves')
        self.label = 0
        self.dataset_splits: Union[List[int], None] = dataset_splits
        self.flux_type: TessFluxType = TessFluxType.PDCSAP

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = TessTwoMinuteCadenceLightcurveMetadata().select()
        query = self.order_by_uuid_with_random_start(query, TessTwoMinuteCadenceLightcurveMetadata.random_order_uuid)
        if self.dataset_splits is not None:
            query = query.where(TessTwoMinuteCadenceLightcurveMetadata.dataset_split.in_(self.dataset_splits))
        return query

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the lightcurve path from the SQL database model.

        :return: The path to the lightcurve.
        """
        return Path(self.tess_two_minute_cadence_lightcurve_metadata_manger.
                    lightcurve_root_directory_path.joinpath(model.path))

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path, self.flux_type)
        return times, fluxes

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the lightcurve/signal file.
        :return: The times and the magnifications of the lightcurve/signal.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path, self.flux_type)
        magnifications, times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return times, magnifications

    def download(self):
        """
        Downloads the lightcurve collection.
        """
        self.tess_data_interface.download_all_two_minute_cadence_lightcurves(self.data_directory)


class TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection(TessTwoMinuteCadenceLightcurveCollection):
    """
    A lightcurve collection of the TESS two minute cadence data with lightcurves from the same target in the same
    dataset split.
    """
    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = TessTwoMinuteCadenceLightcurveMetadata().select()
        query = self.order_by_uuid_with_random_start(query, TessTwoMinuteCadenceLightcurveMetadata.random_order_uuid)
        query = query.join(TessTargetMetadata,
                           on=TessTwoMinuteCadenceLightcurveMetadata.tic_id == TessTargetMetadata.tic_id)
        if self.dataset_splits is not None:
            query = query.where(TessTargetMetadata.dataset_split.in_(self.dataset_splits))
        return query
