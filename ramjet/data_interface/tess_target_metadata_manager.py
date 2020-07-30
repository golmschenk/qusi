"""
Code for managing the metadata of the TESS targets.
"""
from pathlib import Path
from typing import List
from peewee import IntegerField, SchemaManager

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase, convert_class_to_table_name, \
    metadatabase_uuid, dataset_split_from_uuid
from ramjet.data_interface.tess_data_interface import TessDataInterface


class TessTargetMetadata(MetadatabaseModel):
    """
    A model for the TESS target metadatabase table.
    """
    tic_id = IntegerField(index=True, unique=True)
    dataset_split = IntegerField()


class TessTargetMetadataManger:
    """
    A class for managing the metadata of TESS targets.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.lightcurve_root_directory_path = Path('data/tess_two_minute_cadence_lightcurves')

    def insert_multiple_rows_from_paths_into_database(self, lightcurve_paths: List[Path]) -> int:
        """
        Inserts sets targets into the table from lightcurve paths.

        :param lightcurve_paths: The list of paths to insert.
        :return: The number of rows inserted.
        """
        row_dictionary_list = []
        table_name = convert_class_to_table_name(TessTargetMetadata)
        for lightcurve_path in lightcurve_paths:
            tic_id, _ = self.tess_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
            uuid_name = f'{table_name} TIC {tic_id}'
            uuid = metadatabase_uuid(uuid_name)
            dataset_split = dataset_split_from_uuid(uuid)
            row_dictionary_list.append({TessTargetMetadata.tic_id.name: tic_id,
                                        TessTargetMetadata.dataset_split.name: dataset_split})
        with metadatabase.atomic():
            number_of_rows_inserted = TessTargetMetadata.insert_many(row_dictionary_list).on_conflict_ignore().execute()
        return number_of_rows_inserted

    def populate_sql_database(self):
        """
        Populates the SQL database based on the lightcurve files.
        """
        print('Populating the TESS target lightcurve metadata table...')
        path_glob = self.lightcurve_root_directory_path.glob('**/*.fits')
        row_count = 0
        batch_paths = []
        batch_dataset_splits = []
        with metadatabase.atomic():
            for index, path in enumerate(path_glob):
                batch_paths.append(path.relative_to(self.lightcurve_root_directory_path))
                batch_dataset_splits.append(index % 10)
                if index % 1000 == 0 and index != 0:
                    row_count += self.insert_multiple_rows_from_paths_into_database(batch_paths)
                    batch_paths = []
                    batch_dataset_splits = []
                    print(f'{row_count} rows inserted...', end='\r')
            if len(batch_paths) > 0:
                row_count += self.insert_multiple_rows_from_paths_into_database(batch_paths)
        print(f'TESS target metadata table populated. {row_count} rows added.')

    def build_table(self):
        """
        Builds the SQL table.
        """
        TessTargetMetadata.drop_table()
        TessTargetMetadata.create_table()
        self.populate_sql_database()


if __name__ == '__main__':
    manager = TessTargetMetadataManger()
    manager.build_table()
