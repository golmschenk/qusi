"""
Code for managing the meta data of the two minute cadence TESS lightcurves.
"""
from pathlib import Path
from typing import List
from peewee import IntegerField, CharField, SchemaManager

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase, metadatabase_uuid, \
    convert_class_to_table_name, dataset_split_from_uuid
from ramjet.data_interface.tess_data_interface import TessDataInterface


class TessTwoMinuteCadenceLightcurveMetadata(MetadatabaseModel):
    """
    A model for the TESS two minute cadence lightcurve metadatabase table.
    """
    tic_id = IntegerField(index=True)
    sector = IntegerField(index=True)
    path = CharField(unique=True)
    dataset_split = IntegerField()

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('sector', 'tic_id'), True),  # Ensures TIC ID and sector entry is unique.
        )


class TessTwoMinuteCadenceLightcurveMetadataManger:
    """
    A class for managing the meta data of the two minute cadence TESS lightcurves.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.lightcurve_root_directory_path = Path('data/tess_two_minute_cadence_lightcurves')

    def insert_multiple_rows_from_paths_into_database(self, lightcurve_paths: List[Path]):
        """
        Inserts sets of lightcurve paths into the table.

        :param lightcurve_paths: The list of paths to insert.
        """
        row_dictionary_list = []
        table_name = convert_class_to_table_name(TessTwoMinuteCadenceLightcurveMetadata)
        for lightcurve_path in lightcurve_paths:
            tic_id, sector = self.tess_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
            uuid_name = f'{table_name} TIC {tic_id} sector {sector}'
            uuid = metadatabase_uuid(uuid_name)
            dataset_split = dataset_split_from_uuid(uuid)
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
        with metadatabase.atomic():
            for index, path in enumerate(path_glob):
                batch_paths.append(path.relative_to(self.lightcurve_root_directory_path))
                row_count += 1
                if index % 1000 == 0 and index != 0:
                    self.insert_multiple_rows_from_paths_into_database(batch_paths)
                    batch_paths = []
                    print(f'{index} rows inserted...', end='\r')
            if len(batch_paths) > 0:
                self.insert_multiple_rows_from_paths_into_database(batch_paths)
        print(f'TESS two minute cadence lightcurve meta data table populated. {row_count} rows added.')

    def build_table(self):
        """
        Builds the SQL table.
        """
        TessTwoMinuteCadenceLightcurveMetadata.drop_table()
        TessTwoMinuteCadenceLightcurveMetadata.create_table()
        SchemaManager(TessTwoMinuteCadenceLightcurveMetadata).drop_indexes()  # To allow for fast insert.
        self.populate_sql_database()
        print('Building indexes...')
        SchemaManager(TessTwoMinuteCadenceLightcurveMetadata).create_indexes()  # Since we dropped them before.


if __name__ == '__main__':
    manager = TessTwoMinuteCadenceLightcurveMetadataManger()
    manager.build_table()
