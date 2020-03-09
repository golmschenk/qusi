"""Tests for the TessSyntheticInjectedDatabase class."""
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

import ramjet.photometric_database.tess_data_interface
from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase
from tests.picklable_mock import PicklableMock


class TestTessSyntheticInjectedDatabase:
    """Tests for the TessSyntheticInjectedDatabase class."""
    @pytest.fixture
    def database(self) -> TessSyntheticInjectedDatabase:
        """
        Sets up the database for use in a test.

        :return: The database.
        """
        return TessSyntheticInjectedDatabase()

    @pytest.mark.slow
    @pytest.mark.functional
    @patch.object(ramjet.photometric_database.tess_data_interface.fits, 'open')
    @patch.object(ramjet.photometric_database.tess_data_interface.pd, 'read_feather')
    def test_can_generate_training_and_validation_datasets(self, mock_read_feather, mock_fits_open, database):
        # Mock and initialize dataset components for simple testing.
        batch_size = 10
        database.batch_size = batch_size
        time_steps_per_example = 20
        database.time_steps_per_example = time_steps_per_example
        database.lightcurve_directory = PicklableMock(glob=PicklableMock(return_value=(Path(f'{index}.fits')
                                                                                       for index in range(50))))
        database.synthetic_signal_directory = PicklableMock(glob=PicklableMock(return_value=(Path(f'{index}.feather')
                                                                                             for index in range(40))))
        fits_fluxes = np.arange(time_steps_per_example, dtype=np.float32)
        fits_times = fits_fluxes * 10
        hdu = Mock(data={'PDCSAP_FLUX': fits_fluxes, 'TIME': fits_times})
        hdu_list = [None, hdu]  # Lightcurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        synthetic_magnitudes = np.arange(time_steps_per_example + 1)
        synthetic_times = synthetic_magnitudes * 10 * 24  # 24 to make it hours from days.
        mock_read_feather.return_value = pd.DataFrame({'Magnification': synthetic_magnitudes,
                                                       'Time (hours)': synthetic_times})
        # Generate the datasets.
        training_dataset, validation_dataset = database.generate_datasets()
        # Test the datasets look right.
        training_iterator = iter(training_dataset)
        training_batch0 = next(training_iterator)
        training_batch1 = next(training_iterator)
        validation_iterator = iter(validation_dataset)
        validation_batch0 = next(validation_iterator)
        validation_batch1 = next(validation_iterator)
        assert training_batch0[0].shape == (batch_size, time_steps_per_example, 1)  # Batch examples shape
        assert training_batch1[0].shape == (batch_size, time_steps_per_example, 1)
        assert training_batch1[1].shape == (batch_size, 1)  # Batch labels shape
        assert training_batch1[1].numpy().sum() == batch_size // 2  # Half the labels are positive.
        assert validation_batch0[0].shape == (batch_size, time_steps_per_example, 1)
        assert validation_batch1[0].shape == (batch_size, time_steps_per_example, 1)

    @pytest.mark.functional
    @patch.object(ramjet.photometric_database.tess_data_interface.fits, 'open')
    @patch.object(ramjet.photometric_database.tess_data_interface.pd, 'read_feather')
    def test_train_and_validation_preprocessing_produces_an_inject_and_non_injected_lightcurve(self, mock_read_feather,
                                                                                               mock_fits_open,
                                                                                               database):
        # Mock and initialize dataset components for simple testing.
        lightcurve_length = 15
        database.time_steps_per_example = lightcurve_length
        fits_fluxes = np.arange(lightcurve_length, dtype=np.float32)
        fits_times = fits_fluxes * 10
        hdu = Mock(data={'PDCSAP_FLUX': fits_fluxes, 'TIME': fits_times})
        hdu_list = [None, hdu]  # Lightcurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        synthetic_magnitudes = np.arange(16)
        synthetic_times = synthetic_magnitudes * 10 * 24  # 24 to make it hours from days.
        mock_read_feather.return_value = pd.DataFrame({'Magnification': synthetic_magnitudes,
                                                       'Time (hours)': synthetic_times})
        database.number_of_parallel_processes_per_map = 1
        # Generate the datasets.
        examples, labels = database.train_and_validation_preprocessing(tf.convert_to_tensor('fake_path.fits'),
                                                                       tf.convert_to_tensor('fake_path.feather'))
        uninjected_lightcurve, injected_lightcurve = examples
        negative_label, positive_label = labels
        # Test the datasets look right.
        mock_fits_open.assert_called_with('fake_path.fits')
        mock_read_feather.assert_called_with('fake_path.feather')
        assert uninjected_lightcurve.shape == (15, 1)
        assert injected_lightcurve.shape == (15, 1)
        assert negative_label == [0]
        assert positive_label == [1]

    def test_can_inject_signal_into_fluxes(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5])
        lightcurve_times = np.array([10, 20, 30, 40, 50])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                             signal_magnifications, signal_times)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5]))

    def test_flux_preprocessing_gives_a_normalized_correct_length_curve(self, database):
        database.time_steps_per_example = 8
        fluxes = np.array([100, 200, 100, 200, 100, 200])
        preprocessed_fluxes = database.flux_preprocessing(fluxes)
        assert preprocessed_fluxes.shape[0] == 8
        assert np.min(preprocessed_fluxes) > -3
        assert np.max(preprocessed_fluxes) < 3

    def test_inject_signal_errors_on_out_of_bounds(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        lightcurve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with pytest.raises(ValueError):
            database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                   signal_magnifications, signal_times)

    def test_inject_signal_can_be_told_to_allow_out_of_bounds(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        lightcurve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                             signal_magnifications, signal_times,
                                                                             allow_out_of_bounds=True)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))
