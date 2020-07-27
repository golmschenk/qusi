"""
Code representing the collection of TESS two minute cadence lightcurves containing eclipsing binaries.
"""
from typing import Union, List

from peewee import Select

from ramjet.data_interface.tess_eclipsing_binary_metadata_manager import TessEclipsingBinaryMetadata
from ramjet.data_interface.tess_ffi_lightcurve_metadata_manager import TessFfiLightcurveMetadata
from ramjet.photometric_database.derived.tess_ffi_lightcurve_collection import TessFfiLightcurveCollection


class TessFfiEclipsingBinaryLightcurveCollection(TessFfiLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing eclipsing binaries.
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
        eclipsing_binary_tic_id_query = TessEclipsingBinaryMetadata.select(TessEclipsingBinaryMetadata.tic_id)
        query = query.where(TessFfiLightcurveMetadata.tic_id.in_(eclipsing_binary_tic_id_query))
        return query


class TessFfiEclipsingBinaryNegativeLabelLightcurveCollection(TessFfiEclipsingBinaryLightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing eclipsing binaries with a
    negative label.
    """
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 0
