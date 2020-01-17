"""
Tests for the TessDataInterface class.
"""
from typing import Any
from unittest.mock import Mock
import numpy as np
import pandas as pd

from astropy.table import Table

import pytest

from ramjet.photometric_database.tess_data_interface import TessDataInterface


class TestTessDataInterface:
    @pytest.fixture
    def tess_data_interface_module(self) -> Any:
        import ramjet.photometric_database.tess_data_interface as tess_data_interface_module
        return tess_data_interface_module

    @pytest.fixture
    def tess_data_interface(self) ->TessDataInterface:
        return TessDataInterface()

    def test_new_tess_data_interface_sets_astroquery_api_limits(self):
        from astroquery.mast import Observations
        assert Observations.TIMEOUT == 600
        assert Observations.PAGESIZE == 50000
        tess_data_interface = TessDataInterface()
        assert Observations.TIMEOUT == 1200
        assert Observations.PAGESIZE == 10000

    def test_can_getting_time_series_observations_as_pandas_data_frame(self, tess_data_interface,
                                                                       tess_data_interface_module):
        mock_query_result = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.query_criteria = Mock(return_value=mock_query_result)
        query_result = tess_data_interface.get_all_tess_time_series_observations()
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])
