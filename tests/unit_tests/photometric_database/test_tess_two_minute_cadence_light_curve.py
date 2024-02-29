from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

import ramjet.photometric_database.tess_two_minute_cadence_light_curve as module
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import (
    TessMissionLightCurve,
    TessMissionLightCurveColumnName,
    TessMissionLightCurveFitsIndex,
)


class TestTessTwoMinuteCadenceFileBasedLightCurve:
    @pytest.fixture
    def fake_hdu_list(self):
        mock_hdu_data = {
            TessMissionLightCurveFitsIndex.TIME__BTJD.value: np.array(
                [0, 1], dtype=np.float32
            ),
            TessMissionLightCurveFitsIndex.PDCSAP_FLUX.value: np.array(
                [2, 3], dtype=np.float32
            ),
            TessMissionLightCurveFitsIndex.SAP_FLUX.value: np.array(
                [4, 5], dtype=np.float32
            ),
            TessMissionLightCurveFitsIndex.PDCSAP_FLUX_ERROR.value: np.array(
                [6, 7], dtype=np.float32
            ),
            TessMissionLightCurveFitsIndex.SAP_FLUX_ERROR.value: np.array(
                [8, 9], dtype=np.float32
            ),
        }
        mock_hdu = Mock()
        mock_hdu.data = mock_hdu_data
        mock_hdu_list = [
            None,
            mock_hdu,
        ]  # TESS light curve data is in index 1 of the HDU list.
        return mock_hdu_list

    def test_fits_index_enum_have_the_same_entries_as_column_name_enum(self):
        column_name_entry_names = [
            entry.name for entry in TessMissionLightCurveColumnName
        ]
        fits_index_entry_names = [
            entry.name for entry in TessMissionLightCurveFitsIndex
        ]
        assert np.array_equal(
            sorted(column_name_entry_names), sorted(fits_index_entry_names)
        )

    def test_from_path_factory_creates_data_frame_from_fits_hdu_list(
        self, fake_hdu_list
    ):
        with patch.object(module.fits, "open") as mock_open:
            mock_open.return_value.__enter__.return_value = fake_hdu_list
            light_curve = TessMissionLightCurve.from_path(
                Path("TIC 169480782 sector 5.fits")
            )
            expected_data_frame = pd.DataFrame(fake_hdu_list[1].data)
            for column_name, fits_index in zip(
                TessMissionLightCurveColumnName, TessMissionLightCurveFitsIndex
            ):
                assert np.array_equal(
                    light_curve.data_frame[column_name.value],
                    expected_data_frame[fits_index.value],
                )

    def test_from_path_factory_light_curve_uses_correct_default_times_and_fluxes(
        self, fake_hdu_list
    ):
        with patch.object(module.fits, "open") as mock_open:
            mock_open.return_value.__enter__.return_value = fake_hdu_list
            light_curve = TessMissionLightCurve.from_path(
                Path("TIC 169480782 sector 5.fits")
            )
            assert np.array_equal(
                light_curve.times,
                fake_hdu_list[1].data[TessMissionLightCurveFitsIndex.TIME__BTJD.value],
            )
            assert np.array_equal(
                light_curve.fluxes,
                fake_hdu_list[1].data[TessMissionLightCurveFitsIndex.PDCSAP_FLUX.value],
            )

    def test_can_get_tic_id_and_sector_from_human_readable_file_name(self):
        tic_id0, sector0 = TessMissionLightCurve.get_tic_id_and_sector_from_file_path(
            Path("TIC 289890301 sector 15 second half")
        )
        assert tic_id0 == 289890301
        assert sector0 == 15
        tic_id1, sector1 = TessMissionLightCurve.get_tic_id_and_sector_from_file_path(
            Path("output/TIC 169480782 sector 5.png")
        )
        assert tic_id1 == 169480782
        assert sector1 == 5

    def test_get_tic_id_and_sector_raises_error_with_unknown_pattern(self):
        with pytest.raises(
            ValueError,
            match="a b c d e f g does not match a known pattern to extract TIC ID and sector from.",
        ):
            TessMissionLightCurve.get_tic_id_and_sector_from_file_path(
                Path("a b c d e f g")
            )

    def test_can_get_tic_id_and_sector_from_tess_obs_id_style_file_name(self):
        tic_id0, sector0 = TessMissionLightCurve.get_tic_id_and_sector_from_file_path(
            Path(
                "mast:TESS/product/tess2019006130736-s0007-0000000278956474-0131-s_lc.fits"
            )
        )
        assert tic_id0 == 278956474
        assert sector0 == 7
        tic_id1, sector1 = TessMissionLightCurve.get_tic_id_and_sector_from_file_path(
            Path("tess2018319095959-s0005-0000000278956474-0125-s")
        )
        assert tic_id1 == 278956474
        assert sector1 == 5

    def test_from_path_factory_sets_the_tic_id_and_sector_of_the_light_curve(
        self, fake_hdu_list
    ):
        with patch.object(module.fits, "open") as mock_open:
            mock_open.return_value.__enter__.return_value = fake_hdu_list
            light_curve = TessMissionLightCurve.from_path(
                Path("TIC 169480782 sector 5.fits")
            )
            assert light_curve.tic_id == 169480782
            assert light_curve.sector == 5

    @patch.object(module, "download_two_minute_cadence_light_curve")
    def test_from_mast_factory_requests_a_mast_download_and_uses_the_resulting_file_with_the_path_factory(
        self, mock_download
    ):
        mock_light_curve = Mock()
        mock_light_curve_path = Mock()
        mock_download.return_value = mock_light_curve_path
        mock_from_path = Mock(return_value=mock_light_curve)
        TessMissionLightCurve.from_path = mock_from_path
        light_curve = TessMissionLightCurve.from_mast(tic_id=1, sector=2)
        assert mock_download.call_args[1]["tic_id"] == 1
        assert mock_download.call_args[1]["sector"] == 2
        assert mock_from_path.call_args[1]["path"] == mock_light_curve_path
        assert light_curve is mock_light_curve
