import pytest

from ramjet.photometric_database.light_curve_collection import (
    LightCurveCollectionMethodNotImplementedError,
)
from ramjet.photometric_database.sql_metadata_light_curve_collection import (
    SqlMetadataLightCurveCollection,
)


class TestSqlMetadataLightCurveCollection:
    @pytest.fixture
    def collection(self) -> SqlMetadataLightCurveCollection():
        """
        Creates an instance of the class under test.

        :return: An instance of the class under test.
        """
        return SqlMetadataLightCurveCollection()

    def test_get_sql_query_throws_error_when_called_without_implementing(
        self, collection
    ):
        with pytest.raises(LightCurveCollectionMethodNotImplementedError):
            _ = collection.get_sql_query()
