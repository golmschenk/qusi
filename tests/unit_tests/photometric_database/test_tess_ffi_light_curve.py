from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import ramjet.photometric_database.tess_ffi_light_curve as module
from ramjet.photometric_database.tess_ffi_light_curve import (
    TessFfiColumnName,
    TessFfiLightCurve,
    TessFfiPickleIndex,
)


class TestTessFfiDataInterface:
    @pytest.fixture
    def ffi_pickle_contents(
        self,
    ) -> tuple[
        int,
        float,
        float,
        float,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
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
        quality = np.full(10, 0)
        return (
            tic_id,
            ra,
            dec,
            tess_magnitude,
            camera,
            chip,
            time,
            raw_flux,
            corrected_flux,
            pca_flux,
            flux_error,
            quality,
        )

    @patch.object(module.pickle, "load")
    @patch.object(Path, "open")
    def test_can_load_flux_and_data_from_ffi_pickle_files(
        self, mock_open, mock_pickle_load, ffi_pickle_contents
    ):
        light_curve = TessFfiLightCurve()
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path("tesslc_290374453.pkl")
        fluxes, times = light_curve.load_fluxes_and_times_from_pickle_file(
            fake_file_path
        )
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[8])
        assert np.array_equal(times, ffi_pickle_contents[6])

    @patch.object(module.pickle, "load")
    @patch.object(Path, "open")
    def test_can_load_fluxes_flux_errors_and_times_from_ffi_pickle_files(
        self, mock_open, mock_pickle_load, ffi_pickle_contents
    ):
        light_curve = TessFfiLightCurve()
        mock_pickle_load.return_value = ffi_pickle_contents
        fake_file_path = Path("tesslc_290374453.pkl")
        (
            fluxes,
            flux_errors,
            times,
        ) = light_curve.load_fluxes_flux_errors_and_times_from_pickle_file(
            fake_file_path
        )
        assert mock_open.called
        assert np.array_equal(fluxes, ffi_pickle_contents[8])
        assert np.array_equal(flux_errors, ffi_pickle_contents[10])
        assert np.array_equal(times, ffi_pickle_contents[6])

    def test_can_get_tic_id_and_sector_from_ffi_style_file_path(self):
        light_curve = TessFfiLightCurve()
        tic_id0, sector0 = light_curve.get_tic_id_and_sector_from_file_path(
            "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        )
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = light_curve.get_tic_id_and_sector_from_file_path(
            "data/ffi_microlensing_database/light_curves/tesslcs_sector_1/tesslcs_tmag_12_13/tesslc_1234567.pkl"
        )
        assert tic_id1 == 1234567
        assert sector1 == 1
        tic_id2, sector2 = light_curve.get_tic_id_and_sector_from_file_path(
            "tesslc_12345678.pkl"
        )
        assert tic_id2 == 12345678
        assert sector2 is None

    def test_can_get_tic_id_and_sector_from_104_ffi_style_file_path(self):
        light_curve = TessFfiLightCurve()
        tic_id0, sector0 = light_curve.get_tic_id_and_sector_from_file_path(
            "tesslcs_sector_12_104/tesslcs_tmag_1_2/tesslc_290374453"
        )
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = light_curve.get_tic_id_and_sector_from_file_path(
            "data/ffi_microlensing_database/light_curves/tesslcs_sector_1_104/tesslcs_tmag_12_13/tesslc_1234567.pkl"
        )
        assert tic_id1 == 1234567
        assert sector1 == 1
        tic_id2, sector2 = light_curve.get_tic_id_and_sector_from_file_path(
            "tesslc_12345678.pkl"
        )
        assert tic_id2 == 12345678
        assert sector2 is None

    def test_can_get_tic_id_and_sector_from_ffi_two_minute_portion_style_file_path(
        self,
    ):
        light_curve = TessFfiLightCurve()
        tic_id0, sector0 = light_curve.get_tic_id_and_sector_from_file_path(
            "tesslcs_sector_12_104/2_min_cadence_targets/tesslc_290374453"
        )
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = light_curve.get_tic_id_and_sector_from_file_path(
            "data/ffi_microlensing_database/light_curves/tesslcs_sector_1_104/2_min_cadence_targets/tesslc_1234567.pkl"
        )
        assert tic_id1 == 1234567
        assert sector1 == 1
        tic_id2, sector2 = light_curve.get_tic_id_and_sector_from_file_path(
            "tesslc_12345678.pkl"
        )
        assert tic_id2 == 12345678
        assert sector2 is None

    def test_can_get_tic_id_and_sector_from_ffi_project_flat_directory_style_file_path(
        self,
    ):
        light_curve = TessFfiLightCurve()
        tic_id0, sector0 = light_curve.get_tic_id_and_sector_from_file_path(
            "tic_id_290374453_sector_12_ffi_light_curve.pkl"
        )
        assert tic_id0 == 290374453
        assert sector0 == 12
        tic_id1, sector1 = light_curve.get_tic_id_and_sector_from_file_path(
            "data/ffi_microlensing_database/light_curves/tic_id_1234567_sector_1_ffi_light_curve.pkl"
        )
        assert tic_id1 == 1234567
        assert sector1 == 1

    def test_can_get_floor_magnitude_from_ffi_style_file_path(self):
        light_curve = TessFfiLightCurve()
        magnitude0 = light_curve.get_floor_magnitude_from_file_path(
            "tesslcs_sector_12/tesslcs_tmag_2_3/tesslc_290374453"
        )
        assert magnitude0 == 2
        magnitude1 = light_curve.get_floor_magnitude_from_file_path(
            "data/ffi_microlensing_database/light_curves/tesslcs_sector_1/tesslcs_tmag_14_15/tesslc_1234567.pkl"
        )
        assert magnitude1 == 14
        with pytest.raises(
            ValueError,
            match="tesslc_12345678.pkl does not match a known pattern to extract magnitude from.",
        ):
            light_curve.get_floor_magnitude_from_file_path("tesslc_12345678.pkl")

    def test_can_get_floor_magnitude_from_104_ffi_style_file_path(self):
        light_curve = TessFfiLightCurve()
        magnitude0 = light_curve.get_floor_magnitude_from_file_path(
            "tesslcs_sector_12_104/tesslcs_tmag_2_3/tesslc_290374453"
        )
        assert magnitude0 == 2
        magnitude1 = light_curve.get_floor_magnitude_from_file_path(
            "data/ffi_microlensing_database/light_curves/tesslcs_sector_1_104/tesslcs_tmag_14_15/tesslc_1234567.pkl"
        )
        assert magnitude1 == 14
        with pytest.raises(
            ValueError,
            match="tesslc_12345678.pkl does not match a known pattern to extract magnitude from.",
        ):
            light_curve.get_floor_magnitude_from_file_path("tesslc_12345678.pkl")

    def test_all_ffi_column_names_have_matches_in_the_pickle_indexes(self):
        index_names = [index.name for index in TessFfiPickleIndex]
        for column_name in TessFfiColumnName:
            assert column_name.name in index_names

    @patch.object(module.pickle, "load")
    def test_from_path_factory_sets_the_tic_id_and_sector_of_the_light_curve(
        self, mock_pickle_load, ffi_pickle_contents
    ):
        with patch.object(Path, "open"):
            mock_pickle_load.return_value = ffi_pickle_contents
            light_curve = TessFfiLightCurve.from_path(
                Path("tesslcs_sector_1_104/tesslcs_tmag_14_15/tesslc_1234567.pkl")
            )
            assert light_curve.tic_id == 1234567
            assert light_curve.sector == 1
