import itertools
from pathlib import Path
from unittest.mock import Mock

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

    def test_get_paths_paginates_over_sql_query_results(self, collection):
        mock_query = Mock()
        mock_query.paginate = Mock(side_effect=[['0', '1'], ['2', '3'], ['4']])
        collection.get_sql_query = Mock(return_value=mock_query)
        collection.get_path_from_model = lambda model: Path(model)
        paths_generator = collection.get_paths()
        assert list(itertools.islice(paths_generator, 5)) == [Path(str(index)) for index in range(5)]
