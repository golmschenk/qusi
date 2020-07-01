"""
Code for managing the meta data of the two minute cadence TESS lightcurves.
"""
import sqlite3
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import List, Union, Generator
from uuid import uuid4
from peewee import IntegerField, CharField, SchemaManager

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase
from ramjet.data_interface.tess_data_interface import TessDataInterface


class TessTwoMinuteCadenceLightcurveMetadata(MetadatabaseModel):
    """
    A model for the TESS two minute cadence lightcurve metadatabase table.
    """
    tic_id = IntegerField(index=True)
    sector = IntegerField(index=True)
    path = CharField(unique=True)
    dataset_split = IntegerField()
    random_order_uuid = CharField(unique=True, index=True, default=uuid4())

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('dataset_split', 'random_order_uuid'), False),  # Useful for training data split with random order.
            (('tic_id', 'sector'), True),  # Ensures TIC ID and sector entry is unique.
        )


class TessTwoMinuteCadenceLightcurveMetadataManger:
    """
    A class for managing the meta data of the two minute cadence TESS lightcurves.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.database_path = Path('data/metadatabase.sqlite3')
        self.lightcurve_root_directory_path = Path('data/tess_two_minute_cadence_lightcurves')

    def insert_multiple_rows_from_paths_into_database(self, lightcurve_paths: List[Path], dataset_splits: List[int]):
        """
        Inserts sets of lightcurve paths into the table.

        :param lightcurve_paths: The list of paths to insert.
        :param dataset_splits: The dataset splits to assign to each path.
        """
        assert len(lightcurve_paths) == len(dataset_splits)
        row_dictionary_list = []
        for lightcurve_path, dataset_split in zip(lightcurve_paths, dataset_splits):
            tic_id, sector = self.tess_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
            row_dictionary_list.append({TessTwoMinuteCadenceLightcurveMetadata.path.name: str(lightcurve_path),
                                        TessTwoMinuteCadenceLightcurveMetadata.tic_id.name: tic_id,
                                        TessTwoMinuteCadenceLightcurveMetadata.sector.name: sector,
                                        TessTwoMinuteCadenceLightcurveMetadata.dataset_split.name: dataset_split})
        with metadatabase.atomic():
            TessTwoMinuteCadenceLightcurveMetadata.insert_many(row_dictionary_list).execute()

    def populate_sql_database(self):
        """
        Populates the SQL database based on the lightcurve files.
        """
        print('Populating the TESS two minute cadence lightcurve meta data table...')
        path_glob = self.lightcurve_root_directory_path.glob('**/*.fits')
        row_count = 0
        batch_paths = []
        batch_dataset_splits = []
        with metadatabase.atomic():
            for index, path in enumerate(path_glob):
                batch_paths.append(path.relative_to(self.lightcurve_root_directory_path))
                batch_dataset_splits.append(index % 10)
                row_count += 1
                if index % 1000 == 0 and index != 0:
                    self.insert_multiple_rows_from_paths_into_database(batch_paths, batch_dataset_splits)
                    batch_paths = []
                    batch_dataset_splits = []
                    print(f'{index} rows inserted...', end='\r')
            if len(batch_paths) > 0:
                self.insert_multiple_rows_from_paths_into_database(batch_paths, batch_dataset_splits)
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

    def build_table(self):
        """
        Builds the SQL table.
        """
        database_connection_ = sqlite3.connect(self.database_path, uri=True)
        database_cursor_ = database_connection_.cursor()
        TessTwoMinuteCadenceLightcurveMetadata.drop_table()
        TessTwoMinuteCadenceLightcurveMetadata.create_table()
        SchemaManager(TessTwoMinuteCadenceLightcurveMetadata).drop_indexes()  # To allow for fast insert.
        self.populate_sql_database(database_connection_)
        SchemaManager(TessTwoMinuteCadenceLightcurveMetadata).create_indexes()  # Since we dropped them before.

if __name__ == '__main__':
    manager = TessTwoMinuteCadenceLightcurveMetadataManger()
    manager.build_table()
