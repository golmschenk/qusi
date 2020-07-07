"""
Code representing the collection of TESS two minute cadence lightcurves containing transits.
"""
import numpy as np
from pathlib import Path
from typing import List, Union

from peewee import Select

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadata, Disposition
from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import \
    TessTwoMinuteCadenceLightcurveMetadata, TessTwoMinuteCadenceLightcurveMetadataManger
from ramjet.photometric_database.sql_metadata_lightcurve_collection import SqlMetadataLightcurveCollection


class TessTwoMinuteCadenceTransitLightcurveCollection(SqlMetadataLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    tess_data_interface = TessDataInterface()
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()

    def __init__(self, dataset_splits: Union[List[int], None] = None, repeat: bool = True):
        super().__init__(repeat=repeat)
        self.label = 1
        self.dataset_splits: Union[List[int], None] = dataset_splits

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the lightcurve path from the SQL database model.

        :return: The path to the lightcurve.
        """
        return Path(self.tess_two_minute_cadence_lightcurve_metadata_manger.
                    lightcurve_root_directory_path.joinpath(model.path))

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = TessTwoMinuteCadenceLightcurveMetadata().select().order_by(
            TessTwoMinuteCadenceLightcurveMetadata.random_order_uuid)
        if self.dataset_splits is not None:
            query = query.where(TessTwoMinuteCadenceLightcurveMetadata.dataset_split.in_(self.dataset_splits))
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
        query = query.where(TessTwoMinuteCadenceLightcurveMetadata.tic_id.in_(transit_tic_id_query))
        return query

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path)
        return times, fluxes


class TessTwoMinuteCadenceNonTransitLightcurveCollection(SqlMetadataLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    tess_data_interface = TessDataInterface()
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()

    def __init__(self, dataset_splits: Union[List[int], None] = None, repeat=True):
        super().__init__(repeat=repeat)
        self.label = 0
        self.dataset_splits: Union[List[int], None] = dataset_splits

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path)
        return times, fluxes

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the lightcurve path from the SQL database model.

        :return: The path to the lightcurve.
        """
        return Path(self.tess_two_minute_cadence_lightcurve_metadata_manger.
                    lightcurve_root_directory_path.joinpath(model.path))

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = TessTwoMinuteCadenceLightcurveMetadata().select().order_by(
            TessTwoMinuteCadenceLightcurveMetadata.random_order_uuid)
        if self.dataset_splits is not None:
            query = query.where(TessTwoMinuteCadenceLightcurveMetadata.dataset_split.in_(self.dataset_splits))
        transit_candidate_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value |
            TessTransitMetadata.disposition == Disposition.CANDIDATE.value)
        query = query.where(TessTwoMinuteCadenceLightcurveMetadata.tic_id.not_in(transit_candidate_tic_id_query))
        return query