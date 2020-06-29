"""
Code for managing the meta data of the two minute cadence TESS lightcurves.
"""
import sqlite3
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import List, Union, Generator
from uuid import uuid4

from ramjet.data_interface.tess_data_interface import TessDataInterface


class TessTwoMinuteCadenceMetadataManger:
    """
    A class for managing the meta data of the two minute cadence TESS lightcurves.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.database_path = Path('data/meta_database.sqlite3')
        self.lightcurve_root_directory_path = Path('data/tess_two_minute_cadence_lightcurves')

    @staticmethod
    def create_database_table(database_connection: Connection):
        """
        Creates the SQL database table.

        :param database_connection: The database connection to perform the operations on.
        """
        database_cursor = database_connection.cursor()
        database_cursor.execute('''CREATE TABLE TessTwoMinuteCadenceLightcurve (
                                       path TEXT NOT NULL,
                                       tic_id INTEGER NOT NULL,
                                       sector INTEGER NOT NULL,
                                       dataset_split INTEGER NOT NULL,
                                       random_order_uuid TEXT NOT NULL)'''
                                )
        database_connection.commit()

    @staticmethod
    def create_database_table_indexes(database_connection: Connection):
        """
        Creates the indexes for the SQL table.

        :param database_connection: The database connection to perform the operations on.
        """
        database_cursor = database_connection.cursor()
        print('Creating indexes...')
        # Index for the use case of having the entire dataset shuffled.
        database_cursor.execute('''CREATE UNIQUE INDEX TessTwoMinuteCadenceLightcurve_random_order_uuid_index
                                       ON TessTwoMinuteCadenceLightcurve (random_order_uuid)''')
        # Index for the use case of training on the training dataset, then
        # have data shuffled based on the uuid.
        database_cursor.execute('''CREATE INDEX TessTwoMinuteCadenceLightcurve_dataset_split_random_order_uuid_index
                                       ON TessTwoMinuteCadenceLightcurve (dataset_split, random_order_uuid)''')
        # TIC IDs should be fast to index.
        database_cursor.execute('''CREATE INDEX TessTwoMinuteCadenceLightcurve_tic_id_index
                                       ON TessTwoMinuteCadenceLightcurve (tic_id)''')
        # TIC ID and sector together should be unique.
        database_cursor.execute('''CREATE INDEX TessTwoMinuteCadenceLightcurve_tic_id_sector_index
                                               ON TessTwoMinuteCadenceLightcurve (tic_id, sector)''')
        # Paths should be unique.
        database_cursor.execute('''CREATE UNIQUE INDEX TessTwoMinuteCadenceLightcurve_path_index
                                       ON TessTwoMinuteCadenceLightcurve (path)''')
        database_connection.commit()

    def insert_multiple_rows_from_paths_into_database(self, database_cursor: Cursor,
                                                      lightcurve_paths: List[Path],
                                                      dataset_splits: List[int]):
        assert len(lightcurve_paths) == len(dataset_splits)
        sql_values_tuple_list = []
        for lightcurve_path, dataset_split in zip(lightcurve_paths, dataset_splits):
            random_order_uuid = uuid4()
            tic_id, sector = self.tess_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
            sql_values_tuple_list.append((str(lightcurve_path), tic_id, sector, dataset_split, str(random_order_uuid)))
        sql_query_string = f'''INSERT INTO TessTwoMinuteCadenceLightcurve
                                   (path, tic_id, sector, dataset_split, random_order_uuid)
                               VALUES (?, ?, ?, ?, ?)'''
        database_cursor.executemany(sql_query_string, sql_values_tuple_list)

    def populate_sql_database(self, database_connection: Connection):
        """
        Populates the SQL database based on the lightcurve files.
        """
        print('Populating the TESS two minute cadence lightcurve meta data table...')
        database_cursor = database_connection.cursor()
        path_glob = self.lightcurve_root_directory_path.glob('**/*.fits')
        row_count = 0
        batch_paths = []
        batch_dataset_splits = []
        for index, path in enumerate(path_glob):
            batch_paths.append(path.relative_to(self.lightcurve_root_directory_path))
            batch_dataset_splits.append(index % 10)
            row_count += 1
            if index % 1000 == 0 and index != 0:
                self.insert_multiple_rows_from_paths_into_database(database_cursor, batch_paths,
                                                                   batch_dataset_splits)
                batch_paths = []
                batch_dataset_splits = []
                print(f'{index} rows inserted...', end='\r')
        if len(batch_paths) > 0:
            self.insert_multiple_rows_from_paths_into_database(database_cursor, batch_paths,
                                                               batch_dataset_splits)
        database_connection.commit()
        print(f'TESS two minute cadence lightcurve meta data table populated. {row_count} rows added.')

    def create_paths_generator(self, dataset_splits: Union[List[int], None] = None, repeat=True
                               ) -> Generator[Path, None, None]:
        """
        Creates a generator for all the paths from the SQL table, with optional filters.

        :param dataset_splits: The dataset splits to filter on. For splitting training, testing, etc.
        :param repeat: Whether or not the generator should repeat indefinitely.
        :return: The generator.
        """
        batch_size = 1000
        database_cursor = sqlite3.connect(str(self.database_path), uri=True, check_same_thread=False).cursor()
        if dataset_splits is not None:
            dataset_split_condition = f'dataset_split IN ({", ".join(map(str, dataset_splits))})'
        else:
            dataset_split_condition = '1'  # Always true.
        while True:
            database_cursor.execute(f'''SELECT path, random_order_uuid
                                        FROM TessTwoMinuteCadenceLightcurve
                                        WHERE {dataset_split_condition}
                                        ORDER BY random_order_uuid
                                        LIMIT {batch_size}''')
            batch = database_cursor.fetchall()
            while batch:
                for row in batch:
                    yield Path(self.lightcurve_root_directory_path.joinpath(row[0]))
                previous_batch_final_uuid = batch[-1][1]
                database_cursor.execute(f'''SELECT path, random_order_uuid
                                            FROM TessTwoMinuteCadenceLightcurve
                                            WHERE random_order_uuid > '{previous_batch_final_uuid}' AND
                                                  {dataset_split_condition}
                                            ORDER BY random_order_uuid
                                            LIMIT {batch_size}''')
                batch = database_cursor.fetchall()
            if not repeat:
                break
