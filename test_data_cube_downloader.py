"""Tests for the TESS input catalog code."""
import os
import pytest
import numpy as np

from data_cube_downloader import DataCubeDownloader


class TestDataCubeDownloader:
    """Tests for the TESS input catalog code."""

    @pytest.mark.skip
    @pytest.mark.functional
    def test_can_download_a_tess_data_cube_from_a_gaia_source_id(self):
        """Checks that the TESS input catalog can be downloaded."""
        data_cube_downloader = DataCubeDownloader()
        os.path.exists(data_cube_downloader.data_directory)
        gaia_source_id = 4052922352453886976
        expected_tess_input_catalog_id = 50559830
        tess_quarter = 0
        DataCubeDownloader.download_data_cube(gaia_source_id=gaia_source_id)
        tess_data_cube_path = os.path.join(data_cube_downloader.data_directory,
                                           f'{expected_tess_input_catalog_id}_{tess_quarter}.npy')
        assert os.path.exists(tess_data_cube_path)
        tess_data_cube = np.load(tess_data_cube_path)
        assert tess_data_cube.shape[0] == 200
        assert tess_data_cube.shape[1] == 200
        # Add a check for the cube values.
        assert False  # Finish the test!

    @pytest.mark.functional
    def test_can_retrieve_tess_input_catalog_number_from_gaia_source_id(self):
        """Tests that given a Gaia source ID, the TESS input catalog number can be retrieved."""
        data_cube_downloader = DataCubeDownloader()
        gaia_source_id_list = [4612007110183910528, 4612026970112393472]
        expected_tess_input_catalog_id_list = [394780880, 394780842]
        tess_input_catalog_id_list = data_cube_downloader.get_tess_input_catalog_ids_from_gaia_source_ids(
            gaia_source_id_list)
        assert set(expected_tess_input_catalog_id_list) == set(tess_input_catalog_id_list)

    @pytest.mark.functional
    def test_can_retrieve_gaia_cepheid_source_ids(self):
        """Tests that the list of cepheids from the Gaia DR2 can be retrieved."""
        data_cube_downloader = DataCubeDownloader()
        expected_gaia_source_id_list = [5956362046015245056, 261548119462093568]
        unexpected_gaia_source_id_list = [5876134427743626880, 5880735196683220992]
        gaia_source_id_list = data_cube_downloader.get_all_cepheid_gaia_source_ids()
        assert all(entry in gaia_source_id_list for entry in expected_gaia_source_id_list)
        assert not any(entry in gaia_source_id_list for entry in unexpected_gaia_source_id_list)

    @pytest.mark.functional
    def test_can_extract_ffi_image_cube_for_tess_input_catalog_id(self):
        data_cube_downloader = DataCubeDownloader()
        gaia_source_id = 582320496444448896
        cubes = data_cube_downloader.get_data_cubes_for_gaia_source_id(gaia_source_id, cube_side_size=20)
        cube = cubes[0]
        assert cube.shape[0] == 20
        assert cube.shape[1] == 20
