"""
Tests for the TessDataInterface class.
"""
from pathlib import Path
from typing import Any
from unittest.mock import Mock, ANY
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from astropy.table import Table

import pytest

from ramjet.photometric_database.tess_data_interface import TessDataInterface, TessFluxType


class TestTessDataInterface:
    @pytest.fixture
    def tess_data_interface_module(self) -> Any:
        import ramjet.photometric_database.tess_data_interface as tess_data_interface_module
        return tess_data_interface_module

    @pytest.fixture
    def tess_data_interface(self) -> TessDataInterface:
        return TessDataInterface()

    def test_new_tess_data_interface_sets_astroquery_api_limits(self):
        from astroquery.mast import Observations
        assert Observations.TIMEOUT == 600
        assert Observations.PAGESIZE == 50000
        tess_data_interface = TessDataInterface()
        assert Observations.TIMEOUT == 1200
        assert Observations.PAGESIZE == 10000

    def test_can_request_time_series_observations_from_mast_as_pandas_data_frame(self, tess_data_interface,
                                                                                 tess_data_interface_module):
        mock_query_result = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.query_criteria = Mock(return_value=mock_query_result)
        query_result = tess_data_interface.get_all_tess_time_series_observations()
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])

    def test_can_request_data_products_from_mast_as_pandas_data_frame(self, tess_data_interface,
                                                                      tess_data_interface_module):
        mock_query_result = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.get_product_list = Mock(return_value=mock_query_result)
        fake_observations_data_frame = pd.DataFrame()
        query_result = tess_data_interface.get_product_list(fake_observations_data_frame)
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])

    def test_can_request_to_download_products_from_mast(self, tess_data_interface,
                                                        tess_data_interface_module):
        mock_manifest = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.download_products = Mock(return_value=mock_manifest)
        fake_data_products_data_frame = pd.DataFrame()
        data_directory = Path('fake/data/directory')
        query_result = tess_data_interface.download_products(fake_data_products_data_frame,
                                                             data_directory=data_directory)
        tess_data_interface_module.Observations.download_products.assert_called_with(ANY,
                                                                                     download_dir=str(data_directory))
        assert isinstance(query_result, pd.DataFrame)
        assert np.array_equal(query_result['a'].values, [1, 2])

    def test_can_filter_observations_to_get_only_single_sector_observations(self, tess_data_interface):
        observations = pd.DataFrame({'dataURL': ['a_lc.fits', 'b_dvt.fits', 'c_dvr.pdf', 'd_lc.fits']})
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(observations)
        assert single_sector_observations.shape[0] == 2
        assert 'a_lc.fits' in single_sector_observations['dataURL'].values
        assert 'd_lc.fits' in single_sector_observations['dataURL'].values
        assert 'b_dvt.fits' not in single_sector_observations['dataURL'].values

    def test_can_filter_observations_to_get_only_multi_sector_observations(self, tess_data_interface):
        observations = pd.DataFrame({'dataURL': ['a_lc.fits', 'b_dvt.fits', 'c_dvr.pdf', 'd_lc.fits']})
        multi_sector_observations = tess_data_interface.filter_for_multi_sector_observations(observations)
        assert multi_sector_observations.shape[0] == 1
        assert 'b_dvt.fits' in multi_sector_observations['dataURL'].values
        assert 'a_lc.fits' not in multi_sector_observations['dataURL'].values

    def test_can_get_tic_from_single_sector_obs_id(self, tess_data_interface):
        tic_id0 = tess_data_interface.get_tic_id_from_single_sector_obs_id(
            'tess2018206045859-s0001-0000000117544915-0120-s')
        assert tic_id0 == 117544915
        tic_id1 = tess_data_interface.get_tic_id_from_single_sector_obs_id(
            'tess2018319095959-s0005-0000000025132999-0125-s')
        assert tic_id1 == 25132999

    def test_can_get_sector_from_single_sector_obs_id(self, tess_data_interface):
        sector0 = tess_data_interface.get_sector_from_single_sector_obs_id(
            'tess2019112060037-s0011-0000000025132999-0143-s')
        assert sector0 == 11
        sector1 = tess_data_interface.get_sector_from_single_sector_obs_id(
            'tess2018319095959-s0005-0000000025132999-0125-s')
        assert sector1 == 5

    def test_can_add_tic_id_column_to_single_sector_observations(self, tess_data_interface):
        single_sector_observations = pd.DataFrame({'obs_id': ['tess2018319095959-s0005-0000000025132999-0125-s',
                                                              'tess2018206045859-s0001-0000000117544915-0120-s']})
        single_sector_observations = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        assert 'TIC ID' in single_sector_observations.columns
        assert 25132999 in single_sector_observations['TIC ID'].values
        assert 117544915 in single_sector_observations['TIC ID'].values

    def test_can_add_sector_column_to_single_sector_observations(self, tess_data_interface):
        single_sector_observations = pd.DataFrame({'obs_id': ['tess2018319095959-s0005-0000000025132999-0125-s',
                                                              'tess2018206045859-s0001-0000000117544915-0120-s']})
        single_sector_observations = tess_data_interface.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        assert 'Sector' in single_sector_observations.columns
        assert 5 in single_sector_observations['Sector'].values
        assert 1 in single_sector_observations['Sector'].values

    def test_can_load_fluxes_and_times_from_tess_fits(self, tess_data_interface, tess_data_interface_module):
        expected_fluxes = np.array([1, 2, 3], dtype=np.float32)
        expected_times = np.array([4, 5, 6], dtype=np.float32)
        hdu = Mock(data={'PDCSAP_FLUX': expected_fluxes, 'TIME': expected_times})
        hdu_list = [None, hdu]  # Lightcurve information is in first extension table in TESS data.
        tess_data_interface_module.fits.open = Mock(return_value=hdu_list)
        lightcurve_path = 'path/to/lightcurve'
        fluxes, times = tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path)
        tess_data_interface_module.fits.open.assert_called_with(lightcurve_path)
        assert np.array_equal(fluxes, expected_fluxes)
        assert np.array_equal(times, expected_times)

    def test_loading_fluxes_and_times_from_fits_drops_nans(self, tess_data_interface, tess_data_interface_module):
        fits_fluxes = np.array([np.nan, 2, 3], dtype=np.float32)
        fits_times = np.array([4, 5, np.nan], dtype=np.float32)
        hdu = Mock(data={'PDCSAP_FLUX': fits_fluxes, 'TIME': fits_times})
        hdu_list = [None, hdu]  # Lightcurve information is in first extension table in TESS data.
        tess_data_interface_module.fits.open = Mock(return_value=hdu_list)
        lightcurve_path = 'path/to/lightcurve'
        fluxes, times = tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path)
        assert np.array_equal(fluxes, np.array([2], dtype=np.float32))
        assert np.array_equal(times, np.array([5], dtype=np.float32))

    def test_can_extract_different_flux_types_from_fits(self, tess_data_interface, tess_data_interface_module):
        fits_sap_fluxes = np.array([1, 2, 3], dtype=np.float32)
        fits_pdcsap_fluxes = np.array([4, 5, 6], dtype=np.float32)
        fits_times = np.array([7, 8, 9], dtype=np.float32)
        hdu = Mock(data={'SAP_FLUX': fits_sap_fluxes, 'PDCSAP_FLUX': fits_pdcsap_fluxes, 'TIME': fits_times})
        hdu_list = [None, hdu]  # Lightcurve information is in first extension table in TESS data.
        tess_data_interface_module.fits.open = Mock(return_value=hdu_list)
        lightcurve_path = 'path/to/lightcurve'
        sap_fluxes, _ = tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path, TessFluxType.SAP)
        pdcsap_fluxes, _ = tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path,
                                                                                    TessFluxType.PDCSAP)
        assert np.array_equal(sap_fluxes, fits_sap_fluxes)
        assert np.array_equal(pdcsap_fluxes, fits_pdcsap_fluxes)

    def test_can_limit_an_observations_query_by_tic_id(self, tess_data_interface, tess_data_interface_module):
        mock_query_result = Table({'a': [1, 2], 'b': [3, 4]})
        tess_data_interface_module.Observations.query_criteria = Mock(return_value=mock_query_result)
        _ = tess_data_interface.get_all_tess_time_series_observations(tic_id=0)
        assert tess_data_interface_module.Observations.query_criteria.call_args[1]['target_name'] == 0

    def test_can_get_the_coordinates_of_a_target_based_on_tic_id(self, tess_data_interface, tess_data_interface_module):
        mock_query_result = Table({'ra': [62.2, 62.2], 'dec': [-71.4, -71.4]})
        SkyCoord(62.2, -71.4, unit="deg")
        tess_data_interface_module.Catalogs.query_criteria = Mock(return_value=mock_query_result)
        coordinates = tess_data_interface.get_target_coordinates(tic_id=0)
        assert coordinates.ra.deg == 62.2
        assert coordinates.dec.deg == -71.4

