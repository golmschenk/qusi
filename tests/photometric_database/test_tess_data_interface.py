"""
Tests for the TessDataInterface class.
"""
from pathlib import Path
from typing import Any
from unittest.mock import Mock, ANY
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

    def test_can_request_time_series_observations_from_mast_as_pandas_data_frame(self, tess_data_interface,
                                                                                 tess_data_interface_module):
        mock_query_result = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.query_criteria = Mock(return_value=mock_query_result)
        query_result = tess_data_interface.get_all_tess_time_series_observations()
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])

    def test_can_request_data_products_from_mast_as_pandas_data_frame(self, tess_data_interface,
                                                                      tess_data_interface_module):
        mock_query_result = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.get_product_list = Mock(return_value=mock_query_result)
        fake_observations_data_frame = pd.DataFrame()
        query_result = tess_data_interface.get_product_list(fake_observations_data_frame)
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])

    def test_can_request_to_download_products_from_mast(self, tess_data_interface,
                                                        tess_data_interface_module):
        mock_manifest = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.download_products = Mock(return_value=mock_manifest)
        fake_data_products_data_frame = pd.DataFrame()
        data_directory = Path('fake/data/directory')
        query_result = tess_data_interface.download_products(fake_data_products_data_frame,
                                                             data_directory=data_directory)
        tess_data_interface_module.Observations.download_products.assert_called_with(ANY,
                                                                                     download_dir=str(data_directory))
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])
