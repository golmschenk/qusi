from pathlib import Path
from unittest.mock import patch, Mock
import pytest
from peewee import SqliteDatabase

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
        lightcurve_paths = [lightcurve_path0, lightcurve_path1]
        lightcurve_paths = list(map(metadata_manger.tess_ffi_data_interface.lightcurve_root_directory_path.joinpath,
                                    lightcurve_paths))
        mock_get_magnitude_from_file.side_effect = [4.5, 5.5]
        metadata_manger.insert_multiple_rows_from_paths_into_database(
            lightcurve_paths=lightcurve_paths, dataset_splits=[2, 3])
        expected_insert = [
            {'path': str(lightcurve_path0), 'tic_id': 1111, 'sector': 1, 'dataset_split': 2,'magnitude': 7},
            {'path': str(lightcurve_path1), 'tic_id': 1234567, 'sector': 12, 'dataset_split': 3, 'magnitude': 14}
        ]
        assert mock_insert_many.call_args[0][0] == expected_insert

    @patch.object(Path, 'glob')
    def test_can_populate_sql_dataset(self, mock_glob, metadata_manger):
        path_list = [metadata_manger.lightcurve_root_directory_path.joinpath(f'{index}.fits') for index in range(20)]
        mock_glob.return_value = path_list
        mock_insert = Mock()
        metadata_manger.insert_multiple_rows_from_paths_into_database = mock_insert
        metadata_manger.populate_sql_database()
        expected_paths = [Path(f'{index}.fits') for index in range(20)]
        assert mock_insert.call_args[0][0] == path_list
        assert mock_insert.call_args[0][1] == list(range(10)) * 2

    def test_can_retrieve_training_and_validation_path_generator_from_sql_table(self, metadata_manger):
        test_database = SqliteDatabase(':memory:')
        with test_database.bind_ctx([module.TessFfiLightcurveMetadata], bind_refs=False, bind_backrefs=False):
            test_database.connect()
            metadata_manger.build_table()
            metadata_manger.tess_ffi_data_interface.get_tic_id_and_sector_from_file_path = Mock(
                side_effect=zip(range(20), range(20)))
            metadata_manger.tess_ffi_data_interface.get_floor_magnitude_from_file_path = Mock(side_effect=range(20))
            metadata_manger.lightcurve_root_directory_path = Path('')
            lightcurve_paths = [Path(f'{index}.pkl') for index in range(20)]
            dataset_splits = list(range(10)) * 2
            with patch.object(module, 'uuid4') as mock_uuid4:
                mock_uuid4.side_effect = [f'{index:02}' for index in range(20)]
                metadata_manger.insert_multiple_rows_from_paths_into_database(lightcurve_paths, dataset_splits)
            training_data_paths = metadata_manger.create_paths_generator(
                dataset_splits=[0, 1, 2, 3, 4, 5, 6, 7])
            validation_data_paths = metadata_manger.create_paths_generator(
                dataset_splits=[8])
            testing_data_paths = metadata_manger.create_paths_generator(
                dataset_splits=[9])
            training_data_paths_list = list(training_data_paths)
            validation_data_paths_list = list(validation_data_paths)
            testing_data_paths_list = list(testing_data_paths)
            data_directory = metadata_manger.lightcurve_root_directory_path
            assert len(training_data_paths_list) == 16
            assert len(validation_data_paths_list) == 2
            assert len(testing_data_paths_list) == 2
            assert data_directory.joinpath('0.pkl') in training_data_paths_list
            assert data_directory.joinpath('18.pkl') in validation_data_paths_list
            assert data_directory.joinpath('9.pkl') in testing_data_paths_list
            assert len(set(training_data_paths_list).intersection(set(validation_data_paths_list))) == 0
            assert len(set(training_data_paths_list).intersection(set(testing_data_paths_list))) == 0
            assert len(set(validation_data_paths_list).intersection(set(testing_data_paths_list))) == 0
