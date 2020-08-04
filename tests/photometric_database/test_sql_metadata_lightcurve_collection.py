import pytest

from ramjet.photometric_database.lightcurve_collection import LightcurveCollectionMethodNotImplementedError
from ramjet.photometric_database.sql_metadata_lightcurve_collection import SqlMetadataLightcurveCollection


class TestSqlMetadataLightcurveCollection:
    @pytest.fixture
    def collection(self) -> SqlMetadataLightcurveCollection():
        """
        Creates an instance of the class under test.

        :return: An instance of the class under test.
        """
        return SqlMetadataLightcurveCollection()

    def test_get_sql_query_throws_error_when_called_without_implementing(self, collection):
        with pytest.raises(LightcurveCollectionMethodNotImplementedError):
            _ = collection.get_sql_query()
