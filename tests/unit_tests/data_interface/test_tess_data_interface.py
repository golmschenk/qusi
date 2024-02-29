"""
Tests for the TessDataInterface class.
"""
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.utils import TableList

import ramjet.data_interface.tess_data_interface
import ramjet.data_interface.tess_data_interface as tess_data_interface_module
from ramjet.data_interface.tess_data_interface import TessFluxType


class TestTessDataInterface:
    @patch.object(
        ramjet.data_interface.tess_data_interface.Observations, "query_criteria"
    )
    def test_can_request_time_series_observations_from_mast_as_pandas_data_frame(
        self, mock_query_criteria
    ):
        mock_query_result = Table({"a": [1, 2], "b": [3, 4]})
        mock_query_criteria.return_value = mock_query_result
        query_result = (
            tess_data_interface_module.get_all_tess_time_series_observations()
        )
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result["a"].values, [1, 2])

    def test_initialize_astroquery_sets_astroquery_api_limits(self):
        from astroquery.mast import Observations

        Observations.TIMEOUT = 600
        Observations.PAGESIZE = 50000
        tess_data_interface_module.initialize_astroquery()
        assert Observations.TIMEOUT == 2000
        assert Observations.PAGESIZE == 3000

    @patch.object(
        ramjet.data_interface.tess_data_interface.Observations, "get_product_list"
    )
    def test_can_request_data_products_from_mast_as_pandas_data_frame(
        self, mock_get_product_list
    ):
        mock_query_result = Table({"a": [1, 2], "b": [3, 4]})
        mock_get_product_list.return_value = mock_query_result
        fake_observations_data_frame = pd.DataFrame()
        query_result = tess_data_interface_module.get_product_list(
            fake_observations_data_frame
        )
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result["a"].values, [1, 2])

    @patch.object(
        ramjet.data_interface.tess_data_interface.Observations, "download_products"
    )
    def test_can_request_to_download_products_from_mast(self, mock_download_products):
        mock_manifest = Table({"a": [1, 2], "b": [3, 4]})
        mock_download_products.return_value = mock_manifest
        fake_data_products_data_frame = pd.DataFrame()
        data_directory = Path("fake/data/directory")
        query_result = tess_data_interface_module.download_products(
            fake_data_products_data_frame, data_directory=data_directory
        )
        mock_download_products.assert_called_with(ANY, download_dir=str(data_directory))
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result["a"].values, [1, 2])

    def test_can_filter_observations_to_get_only_single_sector_observations(self):
        observations = pd.DataFrame(
            {"dataURL": ["a_lc.fits", "b_dvt.fits", "c_dvr.pdf", "d_lc.fits"]}
        )
        single_sector_observations = (
            tess_data_interface_module.filter_for_single_sector_observations(
                observations
            )
        )
        assert single_sector_observations.shape[0] == 2
        assert "a_lc.fits" in single_sector_observations["dataURL"].values
        assert "d_lc.fits" in single_sector_observations["dataURL"].values
        assert "b_dvt.fits" not in single_sector_observations["dataURL"].values

    def test_can_filter_observations_to_get_only_multi_sector_observations(self):
        observations = pd.DataFrame(
            {"dataURL": ["a_lc.fits", "b_dvt.fits", "c_dvr.pdf", "d_lc.fits"]}
        )
        multi_sector_observations = (
            tess_data_interface_module.filter_for_multi_sector_observations(
                observations
            )
        )
        assert multi_sector_observations.shape[0] == 1
        assert "b_dvt.fits" in multi_sector_observations["dataURL"].values
        assert "a_lc.fits" not in multi_sector_observations["dataURL"].values

    def test_can_get_tic_from_single_sector_obs_id(self):
        tic_id0 = tess_data_interface_module.get_tic_id_from_single_sector_obs_id(
            "tess2018206045859-s0001-0000000117544915-0120-s"
        )
        assert tic_id0 == 117544915
        tic_id1 = tess_data_interface_module.get_tic_id_from_single_sector_obs_id(
            "tess2018319095959-s0005-0000000025132999-0125-s"
        )
        assert tic_id1 == 25132999

    def test_can_get_sector_from_single_sector_obs_id(self):
        sector0 = tess_data_interface_module.get_sector_from_single_sector_obs_id(
            "tess2019112060037-s0011-0000000025132999-0143-s"
        )
        assert sector0 == 11
        sector1 = tess_data_interface_module.get_sector_from_single_sector_obs_id(
            "tess2018319095959-s0005-0000000025132999-0125-s"
        )
        assert sector1 == 5

    def test_can_add_tic_id_column_to_single_sector_observations(self):
        single_sector_observations = pd.DataFrame(
            {
                "obs_id": [
                    "tess2018319095959-s0005-0000000025132999-0125-s",
                    "tess2018206045859-s0001-0000000117544915-0120-s",
                ]
            }
        )
        single_sector_observations = (
            tess_data_interface_module.add_tic_id_column_to_single_sector_observations(
                single_sector_observations
            )
        )
        assert "TIC ID" in single_sector_observations.columns
        assert 25132999 in single_sector_observations["TIC ID"].values
        assert 117544915 in single_sector_observations["TIC ID"].values

    def test_can_add_sector_column_to_single_sector_observations(self):
        single_sector_observations = pd.DataFrame(
            {
                "obs_id": [
                    "tess2018319095959-s0005-0000000025132999-0125-s",
                    "tess2018206045859-s0001-0000000117544915-0120-s",
                ]
            }
        )
        single_sector_observations = (
            tess_data_interface_module.add_sector_column_to_single_sector_observations(
                single_sector_observations
            )
        )
        assert "Sector" in single_sector_observations.columns
        assert 5 in single_sector_observations["Sector"].values
        assert 1 in single_sector_observations["Sector"].values

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_can_load_fluxes_and_times_from_tess_fits(self, mock_fits_open):
        expected_fluxes = np.array([1, 2, 3], dtype=np.float32)
        expected_times = np.array([4, 5, 6], dtype=np.float32)
        hdu = Mock(data={"PDCSAP_FLUX": expected_fluxes, "TIME": expected_times})
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        fluxes, times = tess_data_interface_module.load_fluxes_and_times_from_fits_file(
            light_curve_path
        )
        mock_fits_open.assert_called_with(light_curve_path)
        assert np.array_equal(fluxes, expected_fluxes)
        assert np.array_equal(times, expected_times)

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_loading_fluxes_and_times_from_fits_drops_nans(self, mock_fits_open):
        fits_fluxes = np.array([np.nan, 2, 3], dtype=np.float32)
        fits_times = np.array([4, 5, np.nan], dtype=np.float32)
        hdu = Mock(data={"PDCSAP_FLUX": fits_fluxes, "TIME": fits_times})
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        fluxes, times = tess_data_interface_module.load_fluxes_and_times_from_fits_file(
            light_curve_path
        )
        assert np.array_equal(fluxes, np.array([2], dtype=np.float32))
        assert np.array_equal(times, np.array([5], dtype=np.float32))

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_loading_fluxes_and_times_from_fits_with_nans_kept(self, mock_fits_open):
        fits_fluxes = np.array([np.nan, 2, 3], dtype=np.float32)
        fits_times = np.array([4, 5, np.nan], dtype=np.float32)
        hdu = Mock(data={"PDCSAP_FLUX": fits_fluxes, "TIME": fits_times})
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        fluxes, times = tess_data_interface_module.load_fluxes_and_times_from_fits_file(
            light_curve_path, remove_nans=False
        )
        assert np.allclose(
            fluxes, np.array([np.nan, 2, 3], dtype=np.float32), equal_nan=True
        )
        assert np.allclose(
            times, np.array([4, 5, np.nan], dtype=np.float32), equal_nan=True
        )

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_can_extract_different_flux_types_from_fits(self, mock_fits_open):
        fits_sap_fluxes = np.array([1, 2, 3], dtype=np.float32)
        fits_pdcsap_fluxes = np.array([4, 5, 6], dtype=np.float32)
        fits_times = np.array([7, 8, 9], dtype=np.float32)
        hdu = Mock(
            data={
                "SAP_FLUX": fits_sap_fluxes,
                "PDCSAP_FLUX": fits_pdcsap_fluxes,
                "TIME": fits_times,
            }
        )
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        sap_fluxes, _ = tess_data_interface_module.load_fluxes_and_times_from_fits_file(
            light_curve_path, TessFluxType.SAP
        )
        (
            pdcsap_fluxes,
            _,
        ) = tess_data_interface_module.load_fluxes_and_times_from_fits_file(
            light_curve_path, TessFluxType.PDCSAP
        )
        assert np.array_equal(sap_fluxes, fits_sap_fluxes)
        assert np.array_equal(pdcsap_fluxes, fits_pdcsap_fluxes)

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_can_load_fluxes_flux_errors_and_times_from_tess_fits(self, mock_fits_open):
        expected_fluxes = np.array([1, 2, 3], dtype=np.float32)
        expected_flux_errors = np.array([4, 5, 6], dtype=np.float32)
        expected_times = np.array([7, 8, 9], dtype=np.float32)
        hdu = Mock(
            data={
                "PDCSAP_FLUX": expected_fluxes,
                "PDCSAP_FLUX_ERR": expected_flux_errors,
                "TIME": expected_times,
            }
        )
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        (
            fluxes,
            flux_errors,
            times,
        ) = tess_data_interface_module.load_fluxes_flux_errors_and_times_from_fits_file(
            light_curve_path
        )
        mock_fits_open.assert_called_with(light_curve_path)
        assert np.array_equal(fluxes, expected_fluxes)
        assert np.array_equal(flux_errors, expected_flux_errors)
        assert np.array_equal(times, expected_times)

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_load_fluxes_flux_errors_and_times_from_tess_fits_removing_nans(
        self, mock_fits_open
    ):
        expected_fluxes = np.array([1, 2, np.nan], dtype=np.float32)
        expected_flux_errors = np.array([4, 5, 6], dtype=np.float32)
        expected_times = np.array([np.nan, 8, 9], dtype=np.float32)
        hdu = Mock(
            data={
                "PDCSAP_FLUX": expected_fluxes,
                "PDCSAP_FLUX_ERR": expected_flux_errors,
                "TIME": expected_times,
            }
        )
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        (
            fluxes,
            flux_errors,
            times,
        ) = tess_data_interface_module.load_fluxes_flux_errors_and_times_from_fits_file(
            light_curve_path
        )
        mock_fits_open.assert_called_with(light_curve_path)
        assert np.array_equal(fluxes, [2])
        assert np.array_equal(flux_errors, [5])
        assert np.array_equal(times, [8])

    @patch.object(ramjet.data_interface.tess_data_interface.fits, "open")
    def test_load_fluxes_flux_errors_and_times_from_tess_fits_keeping_nans(
        self, mock_fits_open
    ):
        expected_fluxes = np.array([1, 2, np.nan], dtype=np.float32)
        expected_flux_errors = np.array([4, 5, 6], dtype=np.float32)
        expected_times = np.array([np.nan, 8, 9], dtype=np.float32)
        hdu = Mock(
            data={
                "PDCSAP_FLUX": expected_fluxes,
                "PDCSAP_FLUX_ERR": expected_flux_errors,
                "TIME": expected_times,
            }
        )
        hdu_list = [
            None,
            hdu,
        ]  # LightCurve information is in first extension table in TESS data.
        mock_fits_open.return_value.__enter__.return_value = hdu_list
        light_curve_path = "path/to/light_curve"
        (
            fluxes,
            flux_errors,
            times,
        ) = tess_data_interface_module.load_fluxes_flux_errors_and_times_from_fits_file(
            light_curve_path, remove_nans=False
        )
        mock_fits_open.assert_called_with(light_curve_path)
        assert np.allclose(fluxes, expected_fluxes, equal_nan=True)
        assert np.allclose(flux_errors, expected_flux_errors, equal_nan=True)
        assert np.allclose(times, expected_times, equal_nan=True)

    @patch.object(
        ramjet.data_interface.tess_data_interface.Observations, "query_criteria"
    )
    def test_can_limit_an_observations_query_by_tic_id(self, mock_query_criteria):
        mock_query_result = Table({"a": [1, 2], "b": [3, 4]})
        mock_query_criteria.return_value = mock_query_result
        _ = tess_data_interface_module.get_all_tess_time_series_observations(tic_id=0)
        assert mock_query_criteria.call_args[1]["target_name"] == 0

    @patch.object(ramjet.data_interface.tess_data_interface.Catalogs, "query_criteria")
    def test_can_get_the_coordinates_of_a_target_based_on_tic_id(
        self, mock_query_criteria
    ):
        mock_query_result = Table({"ra": [62.2, 62.2], "dec": [-71.4, -71.4]})
        mock_query_criteria.return_value = mock_query_result
        coordinates = tess_data_interface_module.get_target_coordinates(tic_id=0)
        assert coordinates.ra.deg == 62.2
        assert coordinates.dec.deg == -71.4

    @patch.object(ramjet.data_interface.tess_data_interface.Vizier, "query_region")
    def test_can_get_variable_stars_by_coordinate(self, mock_query_region):
        mock_query_result = TableList({0: Table({"VarType": [b"RR", b"SNI"]})})
        mock_query_region.return_value = mock_query_result
        coordinates = SkyCoord(1, 1, unit="deg")
        variable_data_frame = (
            tess_data_interface_module.get_variable_data_frame_for_coordinates(
                coordinates
            )
        )
        assert variable_data_frame["VarType"].iloc[0] == b"RR"
        assert variable_data_frame["VarType"].iloc[1] == b"SNI"

    @patch.object(ramjet.data_interface.tess_data_interface.Vizier, "query_region")
    def test_returns_empty_data_frame_when_no_variable_stars_exist(
        self, mock_query_region
    ):
        mock_query_result = TableList({})
        mock_query_region.return_value = mock_query_result
        coordinates = SkyCoord(1, 1, unit="deg")
        variable_data_frame = (
            tess_data_interface_module.get_variable_data_frame_for_coordinates(
                coordinates
            )
        )
        assert isinstance(variable_data_frame, pd.DataFrame)

    def test_can_get_variable_stars_by_tic_id(self):
        tic_id_coordinates_result = SkyCoord(62.2, -71.4, unit="deg")
        variables_for_coordinates_result = pd.DataFrame({"VarType": [b"RR", b"SNI"]})
        tess_data_interface_module.get_target_coordinates = Mock(
            return_value=tic_id_coordinates_result
        )
        tess_data_interface_module.get_variable_data_frame_for_coordinates = Mock(
            return_value=variables_for_coordinates_result
        )
        variable_data_frame = (
            tess_data_interface_module.get_variable_data_frame_for_tic_id(tic_id=0)
        )
        assert variable_data_frame["VarType"].iloc[0] == b"RR"
        assert variable_data_frame["VarType"].iloc[1] == b"SNI"

    @patch.object(
        ramjet.data_interface.tess_data_interface.Observations, "query_criteria"
    )
    def test_can_get_list_of_sectors_target_appears_in(self, mock_query):
        mock_query_result = Table(
            {
                "dataURL": [
                    "mast:TESS/product/tess2019006130736-s0007-0000000278956474-0131-s_lc.fits",
                    "mast:TESS/product/tess2018319095959-s0005-0000000278956474-0125-s_lc.fits",
                ],
                "obs_id": [
                    "tess2019006130736-s0007-0000000278956474-0131-s",
                    "tess2018319095959-s0005-0000000278956474-0125-s",
                ],
            }
        )
        mock_query.return_value = mock_query_result
        tic_id = 278956474
        sectors = tess_data_interface_module.get_sectors_target_appears_in(tic_id)
        assert sorted(sectors) == [5, 7]

    def test_can_get_tic_id_and_sector_from_human_readable_file_name(self):
        (
            tic_id0,
            sector0,
        ) = tess_data_interface_module.get_tic_id_and_sector_from_file_path(
            "TIC 289890301 sector 15 second half"
        )
        assert tic_id0 == 289890301
        assert sector0 == 15
        (
            tic_id1,
            sector1,
        ) = tess_data_interface_module.get_tic_id_and_sector_from_file_path(
            "output/TIC 169480782 sector 5.png"
        )
        assert tic_id1 == 169480782
        assert sector1 == 5

    def test_get_tic_id_and_sector_raises_error_with_unknown_pattern(self):
        with pytest.raises(
            ValueError,
            match="a b c d e f g does not match a known pattern to extract "
            "TIC ID and sector from.",
        ):
            tess_data_interface_module.get_tic_id_and_sector_from_file_path(
                "a b c d e f g"
            )

    def test_can_get_tic_id_and_sector_from_tess_obs_id_style_file_name(self):
        (
            tic_id0,
            sector0,
        ) = tess_data_interface_module.get_tic_id_and_sector_from_file_path(
            "mast:TESS/product/tess2019006130736-s0007-0000000278956474-0131-s_lc.fits"
        )
        assert tic_id0 == 278956474
        assert sector0 == 7
        (
            tic_id1,
            sector1,
        ) = tess_data_interface_module.get_tic_id_and_sector_from_file_path(
            "tess2018319095959-s0005-0000000278956474-0125-s"
        )
        assert tic_id1 == 278956474
        assert sector1 == 5

    @patch.object(ramjet.data_interface.tess_data_interface.Catalogs, "query_criteria")
    def test_can_get_the_tic_row_of_a_target_based_on_tic_id(self, mock_query_criteria):
        mock_query_result = Table({"ra": [62.2, 62.2], "dec": [-71.4, -71.4]})
        mock_query_criteria.return_value = mock_query_result
        tic_row = tess_data_interface_module.get_tess_input_catalog_row(tic_id=0)
        assert tic_row["ra"] == 62.2
        assert tic_row["dec"] == -71.4
