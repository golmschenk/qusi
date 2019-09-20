"""Tests for the MicrolensingLabelPerTimeStepDatabase class."""
import pytest
import numpy as np

from photometric_database.microlensing_label_per_time_step_database import MicrolensingLabelPerTimeStepDatabase


class TestMicrolensingLabelPerTimeStepDatabase:
    """Tests for the MicrolensingLabelPerTimeStepDatabase class."""

    @pytest.fixture
    def database(self) -> MicrolensingLabelPerTimeStepDatabase:
        """
        Sets up the database for use in a test.

        :return: The database.
        """
        return MicrolensingLabelPerTimeStepDatabase()

    def test_can_read_microlensing_meta_data_file(self, database):
        meta_data_file_path = 'photometric_database/tests/resources/shortened_candlist_RADec.dat.txt'
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        assert meta_data_frame['tE'].iloc[1] == 1.1946342164440846e+02

    def test_can_calculate_einstein_normalized_separation_in_direction_of_motion(self, database):
        minimum_separation_time0 = np.float32(0)
        observation_time0 = np.float32(100)
        einstein_crossing_time0 = np.float32(200)
        separation0 = database.einstein_normalized_separation_in_direction_of_motion(
            observation_time=observation_time0,
            minimum_separation_time=minimum_separation_time0,
            einstein_crossing_time=einstein_crossing_time0)
        assert separation0 == 1
        minimum_separation_time1 = np.float32(-50)
        observation_time1 = np.float32(-200)
        einstein_crossing_time1 = np.float32(100)
        separation1 = database.einstein_normalized_separation_in_direction_of_motion(
            observation_time=observation_time1,
            minimum_separation_time=minimum_separation_time1,
            einstein_crossing_time=einstein_crossing_time1)
        assert separation1 == -3
        minimum_separation_time2 = np.float32(0)
        observation_times2 = np.float32([1, 2])
        einstein_crossing_time2 = np.float32(1)
        separation2 = database.einstein_normalized_separation_in_direction_of_motion(
            observation_time=observation_times2,
            minimum_separation_time=minimum_separation_time2,
            einstein_crossing_time=einstein_crossing_time2)
        assert np.array_equal(separation2, [2, 4])
