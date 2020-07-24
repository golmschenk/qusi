"""
Code representing the collection of TESS two minute cadence lightcurves containing transits.
"""
from typing import List, Union
from peewee import Select

from ramjet.data_interface.tess_ffi_lightcurve_metadata_manager import TessFfiLightcurveMetadata
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadata, Disposition
from ramjet.photometric_database.derived.tess_ffi_lightcurve_collection import TessFfiLightcurveCollection


class TessFfiConfirmedTransitLightcurveCollection(TessFfiLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
        query = query.where(TessFfiLightcurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessFfiConfirmedAndCandidateTransitLightcurveCollection(TessFfiLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """

    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == Disposition.CONFIRMED.value) |
            (TessTransitMetadata.disposition == Disposition.CANDIDATE.value))
        query = query.where(TessFfiLightcurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessFfiNonTransitLightcurveCollection(TessFfiLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """

    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
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
        query = query.where(TessFfiLightcurveMetadata.tic_id.not_in(transit_candidate_tic_id_query))
        return query
