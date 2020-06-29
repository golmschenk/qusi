"""
Code for managing the meta data of the two minute cadence TESS lightcurves.
"""
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import List
from uuid import uuid4

from ramjet.data_interface.tess_data_interface import TessDataInterface


class TessTwoMinuteCadenceMetaDataManger:
    """
    A class for managing the meta data of the two minute cadence TESS lightcurves.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.database_path = Path('data/meta_database.sqlite3')

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
