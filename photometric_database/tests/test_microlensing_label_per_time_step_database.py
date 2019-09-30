"""Tests for the MicrolensingLabelPerTimeStepDatabase class."""
from pathlib import Path

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

    @pytest.fixture
    def data_directory_path(self) -> str:
        """
        Provides a path to a data directory.

        :return: The data directory path.
        """
        return 'photometric_database/tests/resources/test_data_directory'

    @pytest.fixture
    def positive_directory_path(self) -> str:
        """
        Provides a path to a positive data directory.

        :return: The positive data directory path.
        """
        return 'photometric_database/tests/resources/test_data_directory/positive'

    @pytest.fixture
    def lightcurve_file_path(self, positive_directory_path) -> str:
        """
        Provides a lightcurve file path for the test.

        :return: The lightcurve file path.
        """
        return f'{positive_directory_path}/positive1-R-1-0-100869.phot.cor.feather'

    @pytest.fixture
    def meta_data_file_path(self) -> str:
        """
        Provides a microlensing meta data file path for the test.

        :return: The meta data file path.
        """
        return 'photometric_database/tests/resources/test_candlist_RADec.dat.txt'

    def test_can_read_microlensing_meta_data_file(self, database, meta_data_file_path):
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

    def test_can_get_meta_data_for_lightcurve_file_path(self, database, meta_data_file_path, lightcurve_file_path):
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        lightcurve_meta_data = database.get_meta_data_for_lightcurve_file_path(
            lightcurve_file_path=lightcurve_file_path,
            meta_data_frame=meta_data_frame
        )
        assert lightcurve_meta_data['t0'] == 5000
        assert lightcurve_meta_data['umin'] == -1

    def test_can_generate_magnification_threshold_label_for_lightcurve(self, database, meta_data_file_path,
                                                                       lightcurve_file_path):
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        lightcurve_microlensing_meta_data = database.get_meta_data_for_lightcurve_file_path(lightcurve_file_path,
                                                                                            meta_data_frame)
        times = pd.read_feather(lightcurve_file_path)['HJD'].values
        lightcurve_label = database.magnification_threshold_label_for_lightcurve_meta_data(
            observation_times=times,
            lightcurve_meta_data=lightcurve_microlensing_meta_data,
            threshold=1.1
        )
        expected_label = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        assert np.array_equal(lightcurve_label, expected_label)

    def test_can_calculate_magnifications_for_times(self, database, meta_data_file_path, lightcurve_file_path):
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
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

    def test_can_check_if_meta_data_exists_for_lightcurve_file_path(self, database, meta_data_file_path,
                                                                    lightcurve_file_path, positive_directory_path):
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        has_meta_data0 = database.check_if_meta_data_exists_for_lightcurve_file_path(
            lightcurve_file_path=lightcurve_file_path,
            meta_data_frame=meta_data_frame
        )
        assert has_meta_data0
        has_meta_data1 = database.check_if_meta_data_exists_for_lightcurve_file_path(
            lightcurve_file_path=f'{positive_directory_path}/positive2-R-8-8-8.phot.cor.feather',
            meta_data_frame=meta_data_frame
        )
        assert not has_meta_data1

    def test_can_filter_positive_paths_based_on_available_microlensing_meta_data(self, database,
                                                                                 positive_directory_path,
                                                                                 meta_data_file_path):
        positive_file_paths = list(Path(positive_directory_path).glob('**/*.feather'))
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        assert len(positive_file_paths) == 2
        filtered_positive_file_paths = database.remove_file_paths_with_no_meta_data(file_paths=positive_file_paths,
                                                                                    meta_data_frame=meta_data_frame)
        assert len(filtered_positive_file_paths) == 1

    def test_examples_of_generated_datasets_have_appropriate_sizes(self, database, data_directory_path):
        database.time_steps_per_example = 4
        datasets = database.generate_datasets(
            positive_data_directory=f'{data_directory_path}/positive',
            negative_data_directory=f'{data_directory_path}/negative',
            meta_data_file_path=f'{data_directory_path}/test_candlist_RADec.dat.feather'
        )
        training_dataset, validation_dataset = datasets
        training_batch_examples, training_batch_labels = next(iter(training_dataset))
        assert len(training_batch_examples.shape) == 3
        assert training_batch_examples.shape[1] == 4
        assert training_batch_examples.shape[2] == 2
        assert len(training_batch_labels.shape) == 2
        assert training_batch_labels.shape[1] == 4
        validation_batch_examples, validation_batch_labels = next(iter(validation_dataset))
        assert len(validation_batch_examples.shape) == 3
        assert validation_batch_examples.shape[1] == 10
        assert validation_batch_examples.shape[2] == 2
        assert len(validation_batch_labels.shape) == 2
        assert validation_batch_labels.shape[1] == 10
