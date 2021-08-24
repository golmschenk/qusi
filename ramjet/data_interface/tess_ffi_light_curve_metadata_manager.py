"""
Code for managing the TESS FFI metadata SQL table.
"""
from pathlib import Path
from typing import List
from peewee import IntegerField, CharField, FloatField, SchemaManager

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase, metadatabase_uuid, \
    convert_class_to_table_name, dataset_split_from_uuid
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve


class TessFfiLightCurveMetadata(MetadatabaseModel):
    """
    A model for the TESS FFI light curve metadatabase table.
    """
    tic_id = IntegerField(index=True)
    sector = IntegerField(index=True)
    path = CharField(unique=True)
    dataset_split = IntegerField()
    magnitude = FloatField()

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('sector', 'tic_id'), True),  # Ensures TIC ID and sector entry is unique.
            (('dataset_split', 'tic_id'), False),
            (('dataset_split', 'magnitude', 'tic_id'), False)
        )


class TessFfiLightCurveMetadataManager:
    """
    A class for managing the TESS FFI metadata SQL table.
    """
    def __init__(self):
        self.light_curve_root_directory_path = Path('data/tess_ffi_light_curves')

    def insert_multiple_rows_from_paths_into_database(self, light_curve_paths: List[Path]):
        """
        Inserts sets of light curve paths into the table.

        :param light_curve_paths: The list of paths to insert.
        """
        row_dictionary_list = []
        table_name = convert_class_to_table_name(TessFfiLightCurveMetadata)
        for light_curve_path in light_curve_paths:
            tic_id, sector = TessFfiLightCurve.get_tic_id_and_sector_from_file_path(light_curve_path)
            magnitude = TessFfiLightCurve.get_floor_magnitude_from_file_path(light_curve_path)
            relative_path = light_curve_path.relative_to(self.light_curve_root_directory_path)
            uuid_name = f'{table_name} TIC {tic_id} sector {sector}'
            uuid = metadatabase_uuid(uuid_name)
            dataset_split = dataset_split_from_uuid(uuid)
            row_dictionary_list.append({TessFfiLightCurveMetadata.path.name: str(relative_path),
                                        TessFfiLightCurveMetadata.tic_id.name: tic_id,
                                        TessFfiLightCurveMetadata.sector.name: sector,
                                        TessFfiLightCurveMetadata.magnitude.name: magnitude,
                                        TessFfiLightCurveMetadata.dataset_split.name: dataset_split})
        with metadatabase.atomic():
            TessFfiLightCurveMetadata.insert_many(row_dictionary_list).execute()

    def populate_sql_database(self):
        """
        Populates the SQL database based on the light curve files.
        """
        print('Populating the TESS FFI light curve meta data table...')
        path_glob = self.light_curve_root_directory_path.glob('tesslcs_sector_*_104/tesslcs_tmag_*_*/tesslc_*.pkl')
        row_count = 0
        batch_paths = []
        with metadatabase.atomic():
            for index, path in enumerate(path_glob):
                batch_paths.append(path)
                row_count += 1
                if index % 1000 == 0 and index != 0:
                    self.insert_multiple_rows_from_paths_into_database(batch_paths)
                    batch_paths = []
                    print(f'{index} rows inserted...', end='\r')
            if len(batch_paths) > 0:
                self.insert_multiple_rows_from_paths_into_database(batch_paths)
        print(f'TESS FFI light curve meta data table populated. {row_count} rows added.')

    def build_table(self):
        """
        Builds the SQL table.
        """
        TessFfiLightCurveMetadata.drop_table()
        TessFfiLightCurveMetadata.create_table()
        SchemaManager(TessFfiLightCurveMetadata).drop_indexes()  # To allow for fast insert.
        self.populate_sql_database()
        SchemaManager(TessFfiLightCurveMetadata).create_indexes()  # Since we dropped them before.


if __name__ == '__main__':
    manager = TessFfiLightCurveMetadataManager()
    manager.build_table()
