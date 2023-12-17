"""
Code representing the collection of TESS two minute cadence light curves containing transits.
"""
from typing import List, Union
from peewee import Select

from ramjet.data_interface.tess_ffi_light_curve_metadata_manager import TessFfiLightCurveMetadata
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadata, Disposition
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection


class TessFfiConfirmedTransitLightCurveCollection(TessFfiLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence light curves containing transits.
    """
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessFfiConfirmedAndCandidateTransitLightCurveCollection(TessFfiLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence light curves containing transits.
    """

    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == Disposition.CONFIRMED.value) |
            (TessTransitMetadata.disposition == Disposition.CANDIDATE.value))
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessFfiNonTransitLightCurveCollection(TessFfiLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence light curves containing transits.
    """

    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 0

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_candidate_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == Disposition.CONFIRMED.value) |
            (TessTransitMetadata.disposition == Disposition.CANDIDATE.value))
        query = query.where(TessFfiLightCurveMetadata.tic_id.not_in(transit_candidate_tic_id_query))
        return query
