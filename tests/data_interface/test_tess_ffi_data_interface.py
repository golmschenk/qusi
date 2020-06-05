import sqlite3
from sqlite3 import Connection

import pytest
import numpy as np
import pandas as pd
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
        data_interface = TessFfiDataInterface(database_path=':memory:')
        return data_interface

    @pytest.fixture
    def data_interface_with_sql_rows(self) -> TessFfiDataInterface:
        """
        Sets up the data interfaced for use in a test.

        :return: The data interface.
        """
        data_interface = TessFfiDataInterface(database_path='file::memory:?cache=shared')
        database_connection = sqlite3.connect(data_interface.database_path, uri=True)
        database_cursor = database_connection.cursor()
        data_interface.create_database_lightcurve_table(database_connection)
        data_interface.create_database_lightcurve_table_indexes(database_connection)
        data_interface.get_floor_magnitude_from_file_path = Mock(side_effect=[10, 10, 11, 11, 12] * 4)
        lightcurve_paths = [Path(f'{index}.pkl') for index in range(20)]
        dataset_splits = list(range(10)) * 2
        with patch.object(ramjet.data_interface.tess_ffi_data_interface, 'uuid4') as mock_uuid4:
            mock_uuid4.side_effect = [f'{index:02}' for index in range(20)]
            data_interface.insert_multiple_lightcurve_rows_from_paths_into_database(database_cursor,
                                                                                    lightcurve_paths, dataset_splits)
        database_connection.commit()
        # Keep a link to the database connection to force the shared memory database to persist for the test.
        data_interface.force_memory_database_connection_test_persistence = database_connection
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

    def test_can_glob_lightcurves_by_magnitude(self, data_interface):
        ffi_root_directory = Path('tests/data_interface/test_tess_ffi_data_interface_resources/ffi_lightcurves')
        expected_paths = [
            ffi_root_directory.joinpath('tesslcs_sector_1_104/tesslcs_tmag_12_13/fake0.pkl'),
            ffi_root_directory.joinpath('tesslcs_sector_22_104/tesslcs_tmag_12_13/fake0.pkl')
        ]
        magnitude_filtered_paths = list(data_interface.glob_pickle_path_for_magnitude(ffi_root_directory, 12))
        assert sorted(magnitude_filtered_paths) == sorted(expected_paths)

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

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.sqlite3, 'connect')
    def test_has_a_path_to_lightcurves_directory_with_default(self, mock_connect):
        mock_connect.return_value = Mock(cursor=Mock())
        data_interface0 = TessFfiDataInterface()
        assert data_interface0.lightcurve_root_directory_path == Path('data/tess_ffi_lightcurves')
        data_interface0 = TessFfiDataInterface(lightcurve_root_directory_path=Path('specified/path'))
        assert data_interface0.lightcurve_root_directory_path == Path('specified/path')

    @patch.object(ramjet.data_interface.tess_ffi_data_interface.sqlite3, 'connect')
    def test_has_a_path_to_database_organization_with_default(self, mock_connect):
        mock_connect.return_value = Mock(cursor=Mock())
        data_interface0 = TessFfiDataInterface()
        assert data_interface0.database_path == Path('data/tess_ffi_database.sqlite3')
        data_interface0 = TessFfiDataInterface(database_path=Path('specified/path.sqlite3'))
        assert data_interface0.database_path == Path('specified/path.sqlite3')

    def test_creation_of_database_lightcurve_table_contains_important_columns(self, data_interface):
        database_connection = Connection(data_interface.database_path)
        database_cursor = database_connection.cursor()
        data_interface.create_database_lightcurve_table(database_connection)
        database_cursor.execute('select * from Lightcurve')
        column_names = [description[0] for description in database_cursor.description]
        assert 'path' in column_names
        assert 'magnitude' in column_names
        assert 'dataset_split' in column_names

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

    def test_can_add_sql_database_lightcurve_row_from_path(self, data_interface):
        database_connection = Connection(data_interface.database_path)
        database_cursor = database_connection.cursor()
        data_interface.create_database_lightcurve_table(database_connection)
        lightcurve_path0 = Path('tesslcs_sector_1_104/tesslcs_tmag_7_8/tesslc_1111.pkl')
        uuid0 = 'mock-uuid-output0'
        with patch.object(ramjet.data_interface.tess_ffi_data_interface, 'uuid4') as mock_uuid4:
            mock_uuid4.return_value = uuid0
            data_interface.insert_database_lightcurve_row_from_path(database_cursor, lightcurve_path=lightcurve_path0,
                                                                    dataset_split=2)
        lightcurve_path1 = Path('tesslcs_sector_1_104/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        uuid1 = 'mock-uuid-output1'
        with patch.object(ramjet.data_interface.tess_ffi_data_interface, 'uuid4') as mock_uuid4:
            mock_uuid4.return_value = uuid1
            data_interface.insert_database_lightcurve_row_from_path(database_cursor, lightcurve_path=lightcurve_path1,
                                                                    dataset_split=3)
        database_cursor.execute('SELECT uuid, path, magnitude, dataset_split FROM Lightcurve')
        query_result = database_cursor.fetchall()
        assert query_result == [(uuid0, str(lightcurve_path0), 7, 2),
                                (uuid1, str(lightcurve_path1), 14, 3)]

    def test_indexes_of_sql_database(self, data_interface):
        database_connection = Connection(data_interface.database_path)
        data_interface.create_database_lightcurve_table(database_connection)
        data_interface.create_database_lightcurve_table_indexes(database_connection)
        # noinspection SqlResolve
        results_data_frame = pd.read_sql_query('''SELECT index_list.seq AS index_sequence,
                                                         seqno as index_sequence_number,
                                                         index_info.name as column_name
                                                  FROM pragma_index_list("Lightcurve") index_list,
                                                       pragma_index_info(index_list.name) index_info;''',
                                               database_connection)
        sorted_index_groups = results_data_frame.sort_values('index_sequence_number').groupby('index_sequence')
        column_lists_of_indexes = list(sorted_index_groups['column_name'].apply(list).values)
        assert ['magnitude', 'dataset_split', 'uuid'] in column_lists_of_indexes
        assert ['dataset_split', 'uuid'] in column_lists_of_indexes
        assert ['uuid'] in column_lists_of_indexes

    @patch.object(Path, 'glob')
    def test_can_populate_sql_dataset_from_ffi_directory(self, mock_glob, data_interface):
        database_connection = Connection(data_interface.database_path)
        data_interface.get_floor_magnitude_from_file_path = Mock(return_value=0)
        data_interface.create_database_lightcurve_table(database_connection)
        path_list = [data_interface.lightcurve_root_directory_path.joinpath(f'{index}.pkl') for index in range(20)]
        mock_glob.return_value = path_list
        data_interface.populate_sql_database(database_connection)
        results_data_frame = pd.read_sql_query('SELECT path, dataset_split FROM Lightcurve',
                                               database_connection)
        dataset_split_sizes = results_data_frame.groupby('dataset_split').size()
        assert len(dataset_split_sizes) == 10
        assert all(dataset_split_sizes.values == 2)
        expected_path_string_list = [f'{index}.pkl' for index in range(20)]
        assert sorted(expected_path_string_list) == sorted(list(results_data_frame['path'].values))

    def test_unique_columns_of_sql_table(self, data_interface):
        database_connection = Connection(data_interface.database_path)
        data_interface.create_database_lightcurve_table(database_connection)
        data_interface.create_database_lightcurve_table_indexes(database_connection)
        # noinspection SqlResolve
        results_data_frame = pd.read_sql_query('''SELECT *
                                                  FROM pragma_index_list("Lightcurve") index_list, 
                                                       pragma_index_info(index_list.name) index_info''',
                                               database_connection)
        results_data_frame.columns = ['index_sequence', 'index_name', 'unique', 'origin', 'partial',
                                      'index_sequence_number', 'column_id', 'column_name']
        sorted_index_groups = results_data_frame[results_data_frame['unique'] == 1
                                                 ].sort_values('index_sequence_number').groupby('index_sequence')
        column_lists_of_unique_indexes = list(sorted_index_groups['column_name'].apply(list).values)
        assert ['path'] in column_lists_of_unique_indexes
        assert ['uuid'] in column_lists_of_unique_indexes

    def test_can_insert_multiple_sql_database_lightcurve_rows_from_paths(self, data_interface):
        database_connection = Connection(data_interface.database_path)
        database_cursor = database_connection.cursor()
        data_interface.create_database_lightcurve_table(database_connection)
        lightcurve_path0 = Path('tesslcs_sector_1_104/tesslcs_tmag_7_8/tesslc_1111.pkl')
        lightcurve_path1 = Path('tesslcs_sector_1_104/tesslcs_tmag_14_15/tesslc_1234567.pkl')
        uuid0 = 'mock-uuid-output0'
        uuid1 = 'mock-uuid-output1'
        with patch.object(ramjet.data_interface.tess_ffi_data_interface, 'uuid4') as mock_uuid4:
            mock_uuid4.side_effect = [uuid0, uuid1]
            data_interface.insert_multiple_lightcurve_rows_from_paths_into_database(
                database_cursor, lightcurve_paths=[lightcurve_path0, lightcurve_path1], dataset_splits=[2, 3])
        database_cursor.execute('SELECT uuid, path, magnitude, dataset_split FROM Lightcurve')
        query_result = database_cursor.fetchall()
        assert query_result == [(uuid0, str(lightcurve_path0), 7, 2),
                                (uuid1, str(lightcurve_path1), 14, 3)]

    def test_can_retrieve_training_and_validation_path_generator_from_sql_table(self, data_interface_with_sql_rows):
        training_data_paths = data_interface_with_sql_rows.paths_generator_from_sql_table(
            dataset_splits=[0, 1, 2, 3, 4, 5, 6, 7], repeat=False)
        validation_data_paths = data_interface_with_sql_rows.paths_generator_from_sql_table(
            dataset_splits=[8], repeat=False)
        testing_data_paths = data_interface_with_sql_rows.paths_generator_from_sql_table(
            dataset_splits=[9], repeat=False)
        training_data_paths_list = list(training_data_paths)
        validation_data_paths_list = list(validation_data_paths)
        testing_data_paths_list = list(testing_data_paths)
        data_directory = data_interface_with_sql_rows.lightcurve_root_directory_path
        assert len(training_data_paths_list) == 16
        assert len(validation_data_paths_list) == 2
        assert len(testing_data_paths_list) == 2
        assert data_directory.joinpath('0.pkl') in training_data_paths_list
        assert data_directory.joinpath('18.pkl') in validation_data_paths_list
        assert data_directory.joinpath('9.pkl') in testing_data_paths_list
        assert len(set(training_data_paths_list).intersection(set(validation_data_paths_list))) == 0
        assert len(set(training_data_paths_list).intersection(set(testing_data_paths_list))) == 0
        assert len(set(validation_data_paths_list).intersection(set(testing_data_paths_list))) == 0

    def test_can_retrieve_training_path_generator_filtered_by_magnitude_from_sql_table(
            self, data_interface_with_sql_rows):
        paths_generator = data_interface_with_sql_rows.paths_generator_from_sql_table(
            dataset_splits=[0, 1, 2, 3, 4, 5, 6, 7], magnitudes=[10, 11], repeat=False)
        paths_list = list(paths_generator)
        # 2 of the training dataset split is 12th magnitude.
        assert len(paths_list) == 14
        data_directory = data_interface_with_sql_rows.lightcurve_root_directory_path
        assert data_directory.joinpath('0.pkl') in paths_list
        assert data_directory.joinpath('3.pkl') in paths_list
        assert data_directory.joinpath('4.pkl') not in paths_list  # Excluded because 12th magnitude.
        assert data_directory.joinpath('8.pkl') not in paths_list  # Excluded because dataset split 8.

    def test_can_retrieve_all_paths_generator_from_sql_table(self, data_interface_with_sql_rows):
        paths_generator = data_interface_with_sql_rows.paths_generator_from_sql_table(repeat=False)
        paths_list = list(paths_generator)
        assert len(paths_list) == 20
        data_directory = data_interface_with_sql_rows.lightcurve_root_directory_path
        assert data_directory.joinpath('0.pkl') in paths_list
        assert data_directory.joinpath('3.pkl') in paths_list
        assert data_directory.joinpath('4.pkl') in paths_list
        assert data_directory.joinpath('8.pkl') in paths_list

    def test_all_paths_generator_from_sql_table_can_be_set_to_repeat(self, data_interface_with_sql_rows):
        paths_generator_without_repeat = data_interface_with_sql_rows.paths_generator_from_sql_table(repeat=False)
        with pytest.raises(StopIteration):
            for _ in range(50):
                _ = next(paths_generator_without_repeat)
        paths_generator_with_repeat = data_interface_with_sql_rows.paths_generator_from_sql_table(repeat=True)
        try:
            for _ in range(50):
                _ = next(paths_generator_with_repeat)
        except StopIteration:
            pytest.fail('Generator should have repeated indefinitely, but did not.')

    def test_can_iterate_over_training_and_validation_path_generators_from_sql_table_at_the_same_time(
            self, data_interface_with_sql_rows):
        training_data_path_generator = data_interface_with_sql_rows.paths_generator_from_sql_table(
            dataset_splits=[0, 1, 2, 3, 4, 5, 6, 7])
        validation_data_path_generator = data_interface_with_sql_rows.paths_generator_from_sql_table(
            dataset_splits=[8])
        training_path_list = []
        validation_path_list = []
        for _ in range(6):
            training_path_list.append(next(training_data_path_generator))
            validation_path_list.append(next(validation_data_path_generator))
        data_directory = data_interface_with_sql_rows.lightcurve_root_directory_path
        expected_training_path_list = [data_directory.joinpath(f'{index}.pkl') for index in [0, 1, 2, 3, 4, 5]]
        assert sorted(training_path_list) == sorted(expected_training_path_list)
        expected_validation_path_list = [data_directory.joinpath(f'{index}.pkl') for index in [8, 8, 8, 18, 18, 18]]
        assert sorted(validation_path_list) == sorted(expected_validation_path_list)
