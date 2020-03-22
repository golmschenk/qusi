"""Tests for the FfiToiDatabase class."""
from pathlib import Path
from typing import Tuple
from unittest.mock import patch, Mock
import numpy as np
import pytest

import ramjet.data_interface.tess_data_interface
from ramjet.photometric_database.ffi_toi_database import FfiToiDatabase


class TestFfiToiDatabase:
    """Tests for the FfiToiDatabase class."""

    @pytest.fixture
    def database(self) -> FfiToiDatabase:
        """
        Sets up the database for use in a test.

        :return: The database.
        """
        return FfiToiDatabase()

    @pytest.fixture
    def ffi_pickle_contents(self) -> Tuple[int, float, float, float,
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a mock contents of one of Brian's FFI data files.

        :return: TIC ID, right ascension, declination, TESS magnitude,
                 time, raw flux, corrected flux, PCA flux, flux error.
        """
        tic_id = 231663901
        ra = 62.2
        dec = -71.4
        tess_magnitude = 10
        time = np.arange(0, 100, 10)
        raw_flux = np.arange(10)
        corrected_flux = np.arange(10, 20)
        pca_flux = np.arange(20, 30)
        flux_error = np.arange(0, 1, 0.1)
        return tic_id, ra, dec, tess_magnitude, time, raw_flux, corrected_flux, pca_flux, flux_error

    @patch.object(ramjet.photometric_database.ffi_toi_database.pickle, 'load')
    @patch.object(Path, 'open')
    def test_can_load_flux_and_data_from_ffi_pickle_files(self, mock_open, mock_pickle_load, database,
                                                          ffi_pickle_contents):
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path('fake_path.pkl')
        fluxes, times = database.load_fluxes_and_times_from_ffi_pickle_file(fake_file_path)
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[6])
        assert np.array_equal(times, ffi_pickle_contents[4])

    def test_can_create_synthetic_signal_from_real_data(self, database):
        fluxes = np.array([100, 100, 90, 110, 100, 100])
        times = np.array([100, 110, 120, 130, 140, 150])
        synthetic_magnifications, synthetic_times = database.generate_synthetic_signal_from_real_data(fluxes, times)
        assert np.array_equal(synthetic_magnifications, [1, 1, 0.9, 1.1, 1, 1])
        assert np.array_equal(synthetic_times, [0, 10, 20, 30, 40, 50])

    def test_lightcurve_loading_loads_ffi_data_from_pickle(self, database, ffi_pickle_contents):
        file_fluxes = ffi_pickle_contents[6]
        file_times = ffi_pickle_contents[4]
        file_lightcurve = file_fluxes, file_times
        database.load_fluxes_and_times_from_ffi_pickle_file = Mock(return_value=file_lightcurve)
        fake_file_path = 'fake_path.pkl'
        fluxes, times = database.load_fluxes_and_times_from_lightcurve_path(fake_file_path)
        assert np.array_equal(fluxes, ffi_pickle_contents[6])
        assert np.array_equal(times, ffi_pickle_contents[4])

    def test_synthetic_signal_loading_loads_real_toi_lightcurve_as_synthetic(self, database):
        file_fluxes = np.array([100, 100, 90, 110, 100, 100])
        file_times = np.array([100, 110, 120, 130, 140, 150])
        file_lightcurve = file_fluxes, file_times
        database.tess_data_interface.load_fluxes_and_times_from_fits_file = Mock(return_value=file_lightcurve)
        fake_file_path = 'fake_path.fits'
        magnifications, times = database.load_magnifications_and_times_from_synthetic_signal_path(fake_file_path)
        assert np.array_equal(magnifications, [1, 1, 0.9, 1.1, 1, 1])
        assert np.array_equal(times, [0, 10, 20, 30, 40, 50])

    def test_injecting_out_of_bounds_is_enabled_by_default(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        lightcurve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                             signal_magnifications, signal_times)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))