import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock

from ramjet.data_interface.tess_two_minute_cadence_meta_data_manager import TessTwoMinuteCadenceMetaDataManger
import ramjet.data_interface.tess_two_minute_cadence_meta_data_manager as module


class TestTessTwoMinuteCadenceMetaDataManger:
    @pytest.fixture
    def meta_data_manger(self) -> TessTwoMinuteCadenceMetaDataManger:
        """
        The meta data manager class instance under test.

        :return: The meta data manager.
        """
        return TessTwoMinuteCadenceMetaDataManger()

    def test_can_insert_multiple_sql_database_rows_from_paths(self, meta_data_manger):
        database_cursor = Mock()
        mock_executemany = Mock()
        database_cursor.executemany = mock_executemany
        lightcurve_path0 = Path('lightcurves/tess2019169103026-s0013-0000000382068171-0146-s_lc.fits')
        lightcurve_path1 = Path('lightcurves/tess2019112060037-s0011-0000000280909647-0143-s_lc.fits')
        uuid0 = 'mock-uuid-output0'
        uuid1 = 'mock-uuid-output1'
        with patch.object(module, 'uuid4') as mock_uuid4:
            mock_uuid4.side_effect = [uuid0, uuid1]
            meta_data_manger.insert_multiple_rows_from_paths_into_database(
                database_cursor, lightcurve_paths=[lightcurve_path0, lightcurve_path1], dataset_splits=[2, 3])
        expected_insert_values = [(str(lightcurve_path0), 382068171, 13, 2, uuid0),
                                  (str(lightcurve_path1), 280909647, 11, 3, uuid1)]
        assert mock_executemany.call_args[0][1] == expected_insert_values

    @patch.object(Path, 'glob')
    def test_can_populate_sql_dataset(self, mock_glob, meta_data_manger):
        path_list = [meta_data_manger.lightcurve_root_directory_path.joinpath(f'{index}.fits') for index in range(20)]
        mock_glob.return_value = path_list
        database_connection = Mock()
        mock_insert = Mock()
        meta_data_manger.insert_multiple_rows_from_paths_into_database = mock_insert
        meta_data_manger.populate_sql_database(database_connection)
        assert mock_insert.call_args[0][1] == [Path(f'{index}.fits') for index in range(20)]
        assert mock_insert.call_args[0][2] == list(range(10)) * 2
