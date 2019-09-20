"""Tests for the MicrolensingLabelPerTimeStepDatabase class."""
import pytest

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

    def test_can_get_the_length_of_a_chord_in_a_circle(self, database):
        assert database.length_of_chord_in_circle(radius=50, apothem=5) == pytest.approx(99.4987)
        assert database.length_of_chord_in_circle(radius=50, apothem=25) == pytest.approx(86.6025)
        assert database.length_of_chord_in_circle(radius=50, apothem=45) == pytest.approx(43.5890)
        with pytest.raises(AssertionError):
            database.length_of_chord_in_circle(radius=50, apothem=55)
        with pytest.raises(AssertionError):
            database.length_of_chord_in_circle(radius=50, apothem=-5)

    def test_can_calculate_einstein_normalized_separation_in_direction_of_motion(self, database):
        minimum_separation_time0 = 0
        observation_time0 = 100
        einstein_crossing_time0 = 200
        separation0 = database.einstein_normalized_separation_in_direction_of_motion(
            observation_time=observation_time0,
            minimum_separation_time=minimum_separation_time0,
            einstein_crossing_time=einstein_crossing_time0)
        assert separation0 == 1
        minimum_separation_time1 = -50
        observation_time1 = -200
        einstein_crossing_time1 = 100
        separation1 = database.einstein_normalized_separation_in_direction_of_motion(
            observation_time=observation_time1,
            minimum_separation_time=minimum_separation_time1,
            einstein_crossing_time=einstein_crossing_time1)
        assert separation1 == -3

