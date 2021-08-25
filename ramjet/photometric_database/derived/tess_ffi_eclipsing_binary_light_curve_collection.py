"""
Code representing the collection of TESS two minute cadence light curves containing eclipsing binaries.
"""
from typing import Union, List

from peewee import Select

from ramjet.data_interface.tess_eclipsing_binary_metadata_manager import TessEclipsingBinaryMetadata
from ramjet.data_interface.tess_ffi_light_curve_metadata_manager import TessFfiLightCurveMetadata
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadata, Disposition as TransitDisposition
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection


class TessFfiEclipsingBinaryLightCurveCollection(TessFfiLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence light curves containing eclipsing binaries.
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
        eclipsing_binary_tic_id_query = TessEclipsingBinaryMetadata.select(TessEclipsingBinaryMetadata.tic_id)
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(eclipsing_binary_tic_id_query))
        return query


class TessFfiAntiEclipsingBinaryForTransitLightCurveCollection(TessFfiLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence light curves flagged as eclipsing binaries which are
    not a suspected transit.
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
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == TransitDisposition.CONFIRMED.value) |
            (TessTransitMetadata.disposition == TransitDisposition.CANDIDATE.value))
        eclipsing_binary_tic_id_query = TessEclipsingBinaryMetadata.select(TessEclipsingBinaryMetadata.tic_id).where(
            TessEclipsingBinaryMetadata.tic_id.not_in(transit_tic_id_query))
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(eclipsing_binary_tic_id_query))
        return query
