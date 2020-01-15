"""
Tests for the TessDataInterface class.
"""

import pytest

from ramjet.photometric_database.tess_data_interface import TessDataInterface


class TestTessDataInterface:
    @pytest.fixture
    def test_data_interface(self) -> TessDataInterface:
        return TessDataInterface()

    def test_new_tess_data_interface_sets_astroquery_api_limits(self):
        from astroquery.mast import Observations
        assert Observations.TIMEOUT == 600
        assert Observations.PAGESIZE == 50000
        test_data_interface = TessDataInterface()
        assert Observations.TIMEOUT == 1200
        assert Observations.PAGESIZE == 10000
