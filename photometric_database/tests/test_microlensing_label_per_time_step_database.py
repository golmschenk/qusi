"""Tests for the MicrolensingLabelPerTimeStepDatabase class."""
import pytest
import numpy as np
import pandas as pd

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

    def test_can_calculate_magnification(self, database):
        observation_time0 = np.float32(100)
        minimum_separation_time0 = np.float32(200)
        minimum_einstein_separation0 = np.float32(0)
        einstein_crossing_time0 = np.float32(200)
        magnification0 = database.calculate_magnification(observation_time=observation_time0,
                                                          minimum_separation_time=minimum_separation_time0,
                                                          minimum_einstein_separation=minimum_einstein_separation0,
                                                          einstein_crossing_time=einstein_crossing_time0)
        assert magnification0 == pytest.approx(1.3416407)
        observation_times1 = np.float32([np.sin(np.pi / 4), 0, -1e10])
        minimum_separation_times1 = np.float32(0)
        minimum_einstein_separations1 = np.float32([np.cos(np.pi / 4), 0, 0])
        einstein_crossing_times1 = np.float32(2)
        separations1 = database.calculate_magnification(observation_time=observation_times1,
                                                        minimum_separation_time=minimum_separation_times1,
                                                        minimum_einstein_separation=minimum_einstein_separations1,
                                                        einstein_crossing_time=einstein_crossing_times1)
        assert np.allclose(separations1, [1.3416407, np.inf, 1])

    def test_can_get_meta_data_for_lightcurve_file_path(self, database):
        meta_data_file_path = 'photometric_database/tests/resources/shortened_candlist_RADec.dat.txt'
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        lightcurve_file_path = 'photometric_database/tests/resources/test1-R-1-0-100869.phot.cor.feather'
        lightcurve_meta_data = database.get_meta_data_for_lightcurve_file_path(
            lightcurve_file_path=lightcurve_file_path,
            meta_data_frame=meta_data_frame
        )
        assert lightcurve_meta_data['t0'] == 5000
        assert lightcurve_meta_data['umin'] == -1

    def test_can_calculate_magnification_for_each_lightcurve_time_step(self, database):
        meta_data_file_path = 'photometric_database/tests/resources/shortened_candlist_RADec.dat.txt'
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        lightcurve_file_path = 'photometric_database/tests/resources/test1-R-1-0-100869.phot.cor.feather'
        magnifications = database.calculate_magnifications_for_lightcurve(lightcurve_file_path=lightcurve_file_path,
                                                                          meta_data_frame=meta_data_frame)
        expected_magnifications = [1.22826472, 1.34164079, 1.22826472, 1.10111717, 1.04349839, 1.02015629, 1.01019792,
                                   1.00558664, 1.00327366, 1.00202891]
        assert np.allclose(magnifications, expected_magnifications)

    def test_can_generate_magnification_threshold_label_for_lightcurve(self, database):
        meta_data_file_path = 'photometric_database/tests/resources/shortened_candlist_RADec.dat.txt'
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        lightcurve_file_path = 'photometric_database/tests/resources/test1-R-1-0-100869.phot.cor.feather'
        lightcurve_label = database.magnification_threshold_label_for_lightcurve(
            lightcurve_file_path=lightcurve_file_path,
            meta_data_frame=meta_data_frame,
            threshold=1.1)
        expected_label = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        assert np.array_equal(lightcurve_label, expected_label)

    def test_can_calculate_magnifications_for_times(self, database):
        meta_data_file_path = 'photometric_database/tests/resources/shortened_candlist_RADec.dat.txt'
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        lightcurve_file_path = 'photometric_database/tests/resources/test1-R-1-0-100869.phot.cor.feather'
        lightcurve_microlensing_meta_data = database.get_meta_data_for_lightcurve_file_path(lightcurve_file_path,
                                                                                            meta_data_frame)
        times = pd.read_feather(lightcurve_file_path)['HJD'].values
        magnifications = database.calculate_magnifications_for_lightcurve_meta_data(
            times=times,
            lightcurve_microlensing_meta_data=lightcurve_microlensing_meta_data
        )
        expected_magnifications = [1.22826472, 1.34164079, 1.22826472, 1.10111717, 1.04349839, 1.02015629, 1.01019792,
                                   1.00558664, 1.00327366, 1.00202891]
        assert np.allclose(magnifications, expected_magnifications)
