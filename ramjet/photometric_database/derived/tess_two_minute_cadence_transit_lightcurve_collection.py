"""
Code representing the collection of TESS two minute cadence lightcurves containing transits.
"""
from typing import List, Union
from peewee import Select

from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadata, Disposition
from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import \
    TessTwoMinuteCadenceLightcurveMetadata
from ramjet.photometric_database.derived.tess_two_minute_cadence_lightcurve_collection import \
    TessTwoMinuteCadenceLightcurveCollection


class TessTwoMinuteCadenceTransitLightcurveCollection(TessTwoMinuteCadenceLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    def __init__(self, dataset_splits: Union[List[int], None] = None, repeat: bool = True):
        super().__init__(dataset_splits=dataset_splits, repeat=repeat)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
        query = query.where(TessTwoMinuteCadenceLightcurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessTwoMinuteCadenceNonTransitLightcurveCollection(TessTwoMinuteCadenceLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    def __init__(self, dataset_splits: Union[List[int], None] = None, repeat=True):
        super().__init__(dataset_splits=dataset_splits, repeat=repeat)
        self.label = 0

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_candidate_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == Disposition.CONFIRMED.value) |
            (TessTransitMetadata.disposition == Disposition.CANDIDATE.value))
        query = query.where(TessTwoMinuteCadenceLightcurveMetadata.tic_id.not_in(transit_candidate_tic_id_query))
        return query
