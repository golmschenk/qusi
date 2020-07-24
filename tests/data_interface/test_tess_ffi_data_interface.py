import pytest
import sqlite3
import numpy as np
from uuid import uuid4
from typing import Tuple
from pathlib import Path
from unittest.mock import patch, Mock

import ramjet.data_interface.tess_ffi_data_interface
from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface


class TestTessFfiDataInterface:
    @pytest.fixture
    def data_interface(self) -> TessFfiDataInterface:
        """
        Sets up the data interfaced for use in a test.

        :return: The data interface.
        """
        uuid = uuid4()
        data_interface = TessFfiDataInterface(database_path=f'file:database-{str(uuid)}?mode=memory&cache=shared')
        return data_interface

    @pytest.fixture
    def ffi_pickle_contents(self) -> Tuple[int, float, float, float, int, int, np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray, int]:
        """
        Creates a mock content of one of Brian Powell's FFI data files.

        :return: TIC ID, right ascension, declination, TESS magnitude,
                 time, raw flux, corrected flux, PCA flux, flux error.
        """
        tic_id = 231663901
        ra = 62.2
        dec = -71.4
        tess_magnitude = 10
        camera = 1
        chip = 2
        time = np.arange(0, 100, 10)
        raw_flux = np.arange(10)
        corrected_flux = np.arange(10, 20)
        pca_flux = np.arange(20, 30)
        flux_error = np.arange(0, 1, 0.1)
        quality = 0
        return (tic_id, ra, dec, tess_magnitude, camera, chip, time, raw_flux, corrected_flux, pca_flux, flux_error,
                quality)

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.pickle, 'load')
    @patch.object(Path, 'open')
    def test_can_load_flux_and_data_from_ffi_pickle_files(self, mock_open, mock_pickle_load, data_interface,
                                                          ffi_pickle_contents):
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path('fake_path.pkl')
        fluxes, times = data_interface.load_fluxes_and_times_from_pickle_file(fake_file_path)
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[8])
        assert np.array_equal(times, ffi_pickle_contents[6])

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.pickle, 'load')
    @patch.object(Path, 'open')
    def test_can_load_fluxes_flux_errors_and_times_from_ffi_pickle_files(self, mock_open, mock_pickle_load,
                                                                         data_interface, ffi_pickle_contents):
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path('fake_path.pkl')
        fluxes, flux_errors, times = data_interface.load_fluxes_flux_errors_and_times_from_pickle_file(fake_file_path)
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[8])
        assert np.array_equal(flux_errors, ffi_pickle_contents[10])
        assert np.array_equal(times, ffi_pickle_contents[6])

    def test_can_get_tic_id_and_sector_from_ffi_style_file_path(self, data_interface):
        tic_id0, sector0 = data_interface.get_tic_id_and_sector_from_file_path(
            'tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453')
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = data_interface.get_tic_id_and_sector_from_file_path(
            'data/ffi_microlensing_database/lightcurves/tesslcs_sector_1/tesslcs_tmag_12_13/tesslc_1234567.pkl')
        assert tic_id1 == 1234567
        assert sector1 == 1
        tic_id2, sector2 = data_interface.get_tic_id_and_sector_from_file_path('tesslc_12345678.pkl')
        assert tic_id2 == 12345678
        assert sector2 is None

    def test_can_get_tic_id_and_sector_from_104_ffi_style_file_path(self, data_interface):
        tic_id0, sector0 = data_interface.get_tic_id_and_sector_from_file_path(
            'tesslcs_sector_12_104/tesslcs_tmag_1_2/tesslc_290374453')
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = data_interface.get_tic_id_and_sector_from_file_path(
            'data/ffi_microlensing_database/lightcurves/tesslcs_sector_1_104/tesslcs_tmag_12_13/tesslc_1234567.pkl')
        assert tic_id1 == 1234567
        assert sector1 == 1
        tic_id2, sector2 = data_interface.get_tic_id_and_sector_from_file_path('tesslc_12345678.pkl')
        assert tic_id2 == 12345678
        assert sector2 is None

    def test_can_get_floor_magnitude_from_ffi_style_file_path(self, data_interface):
        magnitude0 = data_interface.get_floor_magnitude_from_file_path(
            'tesslcs_sector_12/tesslcs_tmag_2_3/tesslc_290374453')
        assert magnitude0 == 2
        magnitude1 = data_interface.get_floor_magnitude_from_file_path(
            'data/ffi_microlensing_database/lightcurves/tesslcs_sector_1/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        assert magnitude1 == 14
        with pytest.raises(ValueError):
            data_interface.get_floor_magnitude_from_file_path('tesslc_12345678.pkl')

    def test_can_get_floor_magnitude_from_104_ffi_style_file_path(self, data_interface):
        magnitude0 = data_interface.get_floor_magnitude_from_file_path(
            'tesslcs_sector_12_104/tesslcs_tmag_2_3/tesslc_290374453')
        assert magnitude0 == 2
        magnitude1 = data_interface.get_floor_magnitude_from_file_path(
            'data/ffi_microlensing_database/lightcurves/tesslcs_sector_1_104/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        assert magnitude1 == 14
        with pytest.raises(ValueError):
            data_interface.get_floor_magnitude_from_file_path('tesslc_12345678.pkl')
