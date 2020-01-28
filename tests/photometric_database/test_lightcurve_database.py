"""
Tests for the LightcurveDatabase class.
"""
from typing import Any
from unittest.mock import Mock
import numpy as np
import pytest


from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class TestLightcurveDatabase:
    @pytest.fixture
    def database(self):
        """Fixture of an instance of the class under test."""
        return LightcurveDatabase()

    @pytest.fixture
    def module(self) -> Any:
        """Fixture of the module under test."""
        import ramjet.photometric_database.lightcurve_database as lightcurve_database_module
        return lightcurve_database_module

    def test_extraction_of_chunk_and_remainder_from_array(self, database, module):
        module.np.random.shuffle = Mock()
        array_to_chunk = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        expected_chunk = np.array([[3, 3], [4, 4]])
        expected_remainder = np.array([[1, 1], [2, 2], [5, 5], [6, 6]])
        chunk, remainder = database.extract_shuffled_chunk_and_remainder(array_to_chunk, chunk_ratio=1 / 3,
                                                                         chunk_to_extract_index=1)
        assert np.array_equal(chunk, expected_chunk)
        assert np.array_equal(remainder, expected_remainder)
