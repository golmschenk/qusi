from pathlib import Path
from unittest.mock import patch, Mock

import pytest

import ramjet.data_interface.tess_ffi_lightcurve_metadata_manager as module
from ramjet.data_interface.tess_ffi_lightcurve_metadata_manager import TessFfiLightcurveMetadataManager

class TestTessFfiLightcurveMetadataManager:
    @pytest.fixture
    def metadata_manger(self) -> TessFfiLightcurveMetadataManager:
        """
        The meta data manager class instance under test.

        :return: The meta data manager.
        """
        return TessFfiLightcurveMetadataManager()

    @patch.object(module, 'metadatabase')
    @patch.object(module.TessFfiDataInterface, 'get_magnitude_from_file')
    @patch.object(module.TessFfiLightcurveMetadata, 'insert_many')
    def test_can_insert_multiple_sql_database_rows_from_paths(self, mock_insert_many, mock_get_magnitude_from_file,
                                                              mock_metadatabase, metadata_manger):
        lightcurve_path0 = Path('tesslcs_sector_1_104/tesslcs_tmag_7_8/tesslc_1111.pkl')
        lightcurve_path1 = Path('tesslcs_sector_12_104/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        mock_get_magnitude_from_file.side_effect = [4.5, 5.5]
        metadata_manger.insert_multiple_rows_from_paths_into_database(
            lightcurve_paths=[lightcurve_path0, lightcurve_path1], dataset_splits=[2, 3])
        expected_insert = [
            {'path': str(lightcurve_path0), 'tic_id': 1111, 'sector': 1, 'dataset_split': 2,'magnitude': 4.5},
            {'path': str(lightcurve_path1), 'tic_id': 1234567, 'sector': 12, 'dataset_split': 3, 'magnitude': 5.5}
        ]
        assert mock_insert_many.call_args[0][0] == expected_insert

    @patch.object(Path, 'glob')
    def test_can_populate_sql_dataset(self, mock_glob, metadata_manger):
        path_list = [metadata_manger.lightcurve_root_directory_path.joinpath(f'{index}.fits') for index in range(20)]
        mock_glob.return_value = path_list
        mock_insert = Mock()
        metadata_manger.insert_multiple_rows_from_paths_into_database = mock_insert
        metadata_manger.populate_sql_database()
        assert mock_insert.call_args[0][0] == [Path(f'{index}.fits') for index in range(20)]
        assert mock_insert.call_args[0][1] == list(range(10)) * 2
