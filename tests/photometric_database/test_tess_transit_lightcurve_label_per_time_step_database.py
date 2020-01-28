"""Tests for the TransitLightcurveLabelPerTimeStepDatabase class."""

import pytest
import pandas as pd
from pathlib import Path

from ramjet.photometric_database.tess_transit_lightcurve_label_per_time_step_database import \
    TransitLightcurveLabelPerTimeStepDatabase


class TestTransitLightcurveLabelPerTimeStepDatabase:
    """Tests for the TransitLightcurveLabelPerTimeStepDatabase class."""

    @pytest.fixture
    def data_directory_path(self) -> str:
        """
        Provides a path to a data directory.

        :return: The data directory path.
        """
        return str(Path(__file__).parent.joinpath('resources/test_data_directory/tess'))

    @pytest.fixture
    def database(self, data_directory_path) -> TransitLightcurveLabelPerTimeStepDatabase:
        """
        Sets up the database for use in a test.

        :return: The database.
        """
        return TransitLightcurveLabelPerTimeStepDatabase(data_directory_path)

    def test_can_collect_lightcurve_paths(self, database):
        lightcurve_paths = database.get_lightcurve_file_paths()
        assert len(lightcurve_paths) == 5
        assert any(path.name == 'tess2018206045859-s0001-0000000117544915-0120-s_lc.fits' for path in lightcurve_paths)

    def test_can_determine_if_file_is_positive_based_on_file_path(self, database, data_directory_path):
        database.meta_data_frame = pd.DataFrame(
            {'lightcurve_path': [str(Path(data_directory_path,
                                          'lightcurves/tess2018206045859-s0001-0000000117544915-0120-s_lc.fits'))]}
        )
        example_path0 = str(
            database.lightcurve_directory.joinpath('tess2018206045859-s0001-0000000117544915-0120-s_lc.fits')
        )
        assert database.is_positive(example_path0)
        example_path1 = str(
            database.lightcurve_directory.joinpath('tess2018206045859-s0001-0000000150065151-0120-s_lc.fits')
        )
        assert not database.is_positive(example_path1)
