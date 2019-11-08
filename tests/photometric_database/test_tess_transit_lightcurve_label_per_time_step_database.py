"""Tests for the TessTransitLightcurveLabelPerTimeStepDatabase class."""

import pytest
import pandas as pd
from pathlib import Path

from ramjet.photometric_database.tess_transit_lightcurve_label_per_time_step_database import \
    TessTransitLightcurveLabelPerTimeStepDatabase


class TestTessTransitLightcurveLabelPerTimeStepDatabase:
    """Tests for the TessTransitLightcurveLabelPerTimeStepDatabase class."""

    @pytest.fixture
    def data_directory_path(self) -> str:
        """
        Provides a path to a data directory.

        :return: The data directory path.
        """
        return str(Path(__file__).parent.joinpath('resources/test_data_directory/tess'))

    @pytest.fixture
    def database(self, data_directory_path) -> TessTransitLightcurveLabelPerTimeStepDatabase:
        """
        Sets up the database for use in a test.

        :return: The database.
        """
        return TessTransitLightcurveLabelPerTimeStepDatabase(data_directory_path)

    def test_can_collect_lightcurve_paths(self, database):
        lightcurve_paths = database.get_lightcurve_file_paths()
        assert len(lightcurve_paths) == 5
        assert any(path.name == 'tess2018206045859-s0001-0000000117544915-0120-s_lc.fits' for path in lightcurve_paths)

    def test_can_collect_data_validations_by_tic_id(self, database):
        database.obtain_data_validation_dictionary()
        assert len(database.data_validation_dictionary) == 3
        assert any(path.name == 'tess2018206190142-s0001-s0001-0000000117544915-00106_dvr.xml'
                   for path in database.data_validation_dictionary.values())

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

    def test_can_get_sector_from_single_sector_obs_id(self, database):
        sector0 = database.get_sector_from_single_sector_obs_id('tess2019112060037-s0011-0000000025132999-0143-s')
        assert sector0 == 11
        sector1 = database.get_sector_from_single_sector_obs_id('tess2018319095959-s0005-0000000025132999-0125-s')
        assert sector1 == 5

    def test_can_get_sectors_from_multi_sector_obs_id(self, database):
        start0, end0 = database.get_sectors_from_multi_sector_obs_id(
            'tess2018206190142-s0001-s0009-0000000025132999-00205'
        )
        assert start0 == 1
        assert end0 == 9
        start1, end1 = database.get_sectors_from_multi_sector_obs_id(
            'tess2018206190142-s0002-s0003-0000000025132999-00129'
        )
        assert start1 == 2
        assert end1 == 3
