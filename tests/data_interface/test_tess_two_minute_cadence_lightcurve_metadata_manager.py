import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock

from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import TessTwoMinuteCadenceLightcurveMetadataManger
import ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager as module


class TestTessTwoMinuteCadenceLightcurveMetadataManger:
    @pytest.fixture
    def metadata_manger(self) -> TessTwoMinuteCadenceLightcurveMetadataManger:
        """
        The meta data manager class instance under test.

        :return: The meta data manager.
        """
        return TessTwoMinuteCadenceLightcurveMetadataManger()

    @patch.object(module, 'metadatabase')
    @patch.object(module.TessTwoMinuteCadenceLightcurveMetadata, 'insert_many')
    def test_can_insert_multiple_sql_database_rows_from_paths(self, mock_insert_many, mock_metadatabase,
                                                              metadata_manger):
        lightcurve_path0 = Path('lightcurves/tess2019169103026-s0013-0000000382068171-0146-s_lc.fits')
        lightcurve_path1 = Path('lightcurves/tess2019112060037-s0011-0000000280909647-0143-s_lc.fits')
        uuid0 = 'mock-uuid-output0'
        uuid1 = 'mock-uuid-output1'
        with patch.object(module, 'metadatabase_uuid') as mock_metadatabase_uuid:
            with patch.object(module, 'dataset_split_from_uuid') as mock_dataset_split_generator:
                mock_dataset_split_generator.side_effect = [2, 3]
                mock_metadatabase_uuid.side_effect = [uuid0, uuid1]
                metadata_manger.insert_multiple_rows_from_paths_into_database(
                    lightcurve_paths=[lightcurve_path0, lightcurve_path1])
        expected_insert = [{'random_order_uuid': 'mock-uuid-output0', 'path': str(lightcurve_path0),
                            'tic_id': 382068171, 'sector': 13, 'dataset_split': 2},
                           {'random_order_uuid': 'mock-uuid-output1', 'path': str(lightcurve_path1),
                            'tic_id': 280909647, 'sector': 11, 'dataset_split': 3}]
        assert mock_insert_many.call_args[0][0] == expected_insert

    @patch.object(Path, 'glob')
    def test_can_populate_sql_dataset(self, mock_glob, metadata_manger):
        path_list = [metadata_manger.lightcurve_root_directory_path.joinpath(f'{index}.fits') for index in range(20)]
        mock_glob.return_value = path_list
        mock_insert = Mock()
        metadata_manger.insert_multiple_rows_from_paths_into_database = mock_insert
        metadata_manger.populate_sql_database()
        assert mock_insert.call_args[0][0] == [Path(f'{index}.fits') for index in range(20)]
