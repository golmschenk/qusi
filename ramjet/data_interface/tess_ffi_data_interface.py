"""
Code for interfacing with Brian Powell's TESS full frame image (FFI) data.
"""
import re
import pickle
import sqlite3
import numpy as np
from uuid import uuid4
from enum import Enum
from pathlib import Path
from sqlite3 import Cursor, Connection
from typing import Union, List, Iterable, Generator


class FfiDataIndexes(Enum):
    """
    An enum for accessing Brian Powell's FFI pickle data with understandable indexes.
    """
    TIC_ID = 0
    RA = 1
    DEC = 2
    TESS_MAGNITUDE = 3
    CAMERA = 4
    CHIP = 5
    TIME__BTJD = 6
    RAW_FLUX = 7
    CORRECTED_FLUX = 8
    PCA_FLUX = 9
    FLUX_ERROR = 10
    QUALITY = 11


class TessFfiDataInterface:
    """
    A class for interfacing with Brian Powell's TESS full frame image (FFI) data.
    """

    def __init__(self, lightcurve_root_directory_path: Path = Path('data/tess_ffi_lightcurves'),
                 database_path: Union[Path, str] = Path('data/tess_ffi_database.sqlite3')):
        self.lightcurve_root_directory_path: Path = lightcurve_root_directory_path
        self.database_path: Union[Path, str] = database_path

    @staticmethod
    def load_fluxes_and_times_from_pickle_file(file_path: Union[Path, str],
                                               flux_type_index: FfiDataIndexes = FfiDataIndexes.CORRECTED_FLUX
                                               ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_type_index: The flux type to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[flux_type_index.value]
        times = lightcurve[FfiDataIndexes.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    @staticmethod
    def glob_pickle_path_for_magnitude(ffi_root_directory: Path, magnitude: int) -> Iterable[Path]:
        return ffi_root_directory.glob(f'tesslcs_sector_*_104/tesslcs_tmag_{magnitude}_{magnitude + 1}/*.pkl')

    @staticmethod
    def load_fluxes_flux_errors_and_times_from_pickle_file(
            file_path: Union[Path, str], flux_type_index: FfiDataIndexes = FfiDataIndexes.CORRECTED_FLUX
    ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes, flux errors, and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_type_index: The flux type to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[flux_type_index.value]
        flux_errors = lightcurve[FfiDataIndexes.FLUX_ERROR.value]
        times = lightcurve[FfiDataIndexes.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        assert times.shape == flux_errors.shape
        return fluxes, flux_errors, times

    @staticmethod
    def get_tic_id_and_sector_from_file_path(file_path: Union[Path, str]) -> (int, int):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param file_path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_(\d+)(?:_104)?/tesslcs_tmag_\d+_\d+/tesslc_(\d+)', file_path)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Search for Brian Powell's FFI path convention with only the file name containing the target.
        # E.g., "tesslc_290374453"
        match = re.search(r'tesslc_(\d+)', file_path)
        if match:
            return int(match.group(1)), None
        # Raise an error if none of the patterns matched.
        raise ValueError(f'{file_path} does not match a known pattern to extract TIC ID and sector from.')

    @staticmethod
    def get_floor_magnitude_from_file_path(file_path: Union[Path, str]) -> int:
        """
        Gets the floor magnitude from the FFI file path.

        :param file_path: The path of the file to extract the magnitude.
        :return: The magnitude floored.
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_\d+(?:_104)?/tesslcs_tmag_(\d+)_\d+/tesslc_\d+', file_path)
        if match:
            return int(match.group(1))
        raise ValueError(f'{file_path} does not match a known pattern to extract magnitude from.')

    @staticmethod
    def create_database_lightcurve_table(database_connection: Connection):
        """
        Creates the SQL database table for the FFI dataset, with indexes.

        :param database_connection: The database connection to perform the operations on.
        """
        database_cursor = database_connection.cursor()
        database_cursor.execute('''CREATE TABLE TessFfiLightcurve (
                                       random_order_uuid TEXT NOT NULL,
                                       path TEXT NOT NULL,
                                       magnitude INTEGER NOT NULL,
                                       dataset_split INTEGER NOT NULL)'''
                                )
        database_connection.commit()

    @staticmethod
    def create_database_lightcurve_table_indexes(database_connection: Connection):
        """
        Creates the indexes for the SQL table.

        :param database_connection: The database connection to perform the operations on.
        """
        database_cursor = database_connection.cursor()
        print('Creating indexes...')
        # Index for the use case of having the entire dataset shuffled.
        database_cursor.execute('''CREATE UNIQUE INDEX Lightcurve_random_order_uuid_index
                                        ON TessFfiLightcurve (random_order_uuid)''')
        # Index for the use case of training on the entire dataset, get the training dataset, then
        # have data shuffled based on the uuid.
        database_cursor.execute('''CREATE INDEX Lightcurve_dataset_split_random_order_uuid_index
                                        ON TessFfiLightcurve (dataset_split, random_order_uuid)''')
        # Index for the use case of training on a specific magnitude, get the training dataset, then
        # have data shuffled based on the uuid.
        database_cursor.execute('''CREATE INDEX Lightcurve_magnitude_dataset_split_random_order_uuid_index
                                        ON TessFfiLightcurve (magnitude, dataset_split, random_order_uuid)''')
        # Paths should be unique.
        database_cursor.execute('''CREATE UNIQUE INDEX Lightcurve_path_index
                                        ON TessFfiLightcurve (path)''')
        database_connection.commit()

    def insert_database_lightcurve_row_from_path(self, database_cursor: Cursor, lightcurve_path: Path,
                                                 dataset_split: int):
        """
        Inserts the given lightcurve path into the SQL database with a dataset split tag.

        :param database_cursor: The cursor of the database to perform the operations on.
        :param lightcurve_path: The path of the lightcurve to be added.
        :param dataset_split: The dataset split this path belongs to. That is, an integer which can be used to split
                              the data into training, validation, and testing. Most often 0-9 to provide 10% splits.
        """
        uuid = uuid4()
        magnitude = self.get_floor_magnitude_from_file_path(lightcurve_path)
        database_cursor.execute(
            f'''INSERT INTO TessFfiLightcurve (random_order_uuid, path, magnitude, dataset_split)
                VALUES ('{str(uuid)}', '{str(lightcurve_path)}', {magnitude}, {dataset_split})'''
        )

    def insert_multiple_lightcurve_rows_from_paths_into_database(self, database_cursor: Cursor,
                                                                 lightcurve_paths: List[Path],
                                                                 dataset_splits: List[int]):
        assert len(lightcurve_paths) == len(dataset_splits)
        sql_values_tuple_list = []
        for lightcurve_path, dataset_split in zip(lightcurve_paths, dataset_splits):
            uuid = uuid4()
            magnitude = self.get_floor_magnitude_from_file_path(lightcurve_path)
            sql_values_tuple_list.append((str(uuid), str(lightcurve_path), magnitude, dataset_split))

        sql_query_string = f'''INSERT INTO TessFfiLightcurve (random_order_uuid, path, magnitude, dataset_split)
                               VALUES (?, ?, ? ,?)'''
        database_cursor.executemany(sql_query_string, sql_values_tuple_list)

    def populate_sql_database(self, database_connection: Connection):
        """
        Populates the SQL database based on the files found in the root FFI data directory.
        """
        print('Populating TESS FFI SQL database (this may take a while)...')
        database_cursor = database_connection.cursor()
        path_glob = self.lightcurve_root_directory_path.glob('tesslcs_sector_*_104/tesslcs_tmag_*_*/tesslc_*.pkl')
        row_count = 0
        batch_paths = []
        batch_dataset_splits = []
        for index, path in enumerate(path_glob):
            batch_paths.append(path.relative_to(self.lightcurve_root_directory_path))
            batch_dataset_splits.append(index % 10)
            row_count += 1
            if index % 1000 == 0:
                self.insert_multiple_lightcurve_rows_from_paths_into_database(database_cursor, batch_paths,
                                                                              batch_dataset_splits)
                batch_paths = []
                batch_dataset_splits = []
                print(f'{index} rows inserted...', end='\r')
        if len(batch_paths) > 0:
            self.insert_multiple_lightcurve_rows_from_paths_into_database(database_cursor, batch_paths,
                                                                          batch_dataset_splits)
        database_connection.commit()
        print(f'TESS FFI SQL database populated. {row_count} rows added.')

    def paths_generator_from_sql_table(self, dataset_splits: Union[List[int], None] = None,
                                       magnitudes: Union[List[int], None] = None, repeat=True
                                       ) -> Generator[Path, None, None]:
        """
        Creates a generator for all the paths from the SQL table, with optional filters.

        :param dataset_splits: The dataset splits to filter on. For splitting training, testing, etc.
        :param magnitudes: The target floor magnitudes to filter on.
        :param repeat: Whether or not the generator should repeat indefinitely.
        :return: The generator.
        """
        batch_size = 1000
        database_cursor = sqlite3.connect(str(self.database_path), uri=True, check_same_thread=False).cursor()
        if dataset_splits is not None:
            dataset_split_condition = f'dataset_split IN ({", ".join(map(str, dataset_splits))})'
        else:
            dataset_split_condition = '1'  # Always true.
        if magnitudes is not None:
            magnitude_condition = f'magnitude IN ({", ".join(map(str, magnitudes))})'
        else:
            magnitude_condition = '1'  # Always true.
        while True:
            database_cursor.execute(f'''SELECT path, random_order_uuid
                                        FROM TessFfiLightcurve
                                        WHERE {dataset_split_condition} AND
                                              {magnitude_condition}
                                        ORDER BY random_order_uuid
                                        LIMIT {batch_size}''')
            batch = database_cursor.fetchall()
            while batch:
                for row in batch:
                    yield Path(self.lightcurve_root_directory_path.joinpath(row[0]))
                previous_batch_final_uuid = batch[-1][1]
                database_cursor.execute(f'''SELECT path, random_order_uuid
                                            FROM TessFfiLightcurve
                                            WHERE random_order_uuid > '{previous_batch_final_uuid}' AND
                                                  {dataset_split_condition} AND
                                                  {magnitude_condition}
                                            ORDER BY random_order_uuid
                                            LIMIT {batch_size}''' )
                batch = database_cursor.fetchall()
            if not repeat:
                break


if __name__ == '__main__':
    tess_ffi_data_interface = TessFfiDataInterface()
    database_connection_ = sqlite3.connect(tess_ffi_data_interface.database_path, uri=True)
    database_cursor_ = database_connection_.cursor()
    database_cursor_.execute(f'PRAGMA cache_size = -{2e6}')  # Set the cache size to 2GB.
    database_connection_.commit()
    database_cursor_.execute('DROP TABLE IF EXISTS TessFfiLightcurve')
    database_connection_.commit()
    tess_ffi_data_interface.create_database_lightcurve_table(database_connection_)
    tess_ffi_data_interface.populate_sql_database(database_connection_)
    tess_ffi_data_interface.create_database_lightcurve_table_indexes(database_connection_)
    database_connection_.close()
