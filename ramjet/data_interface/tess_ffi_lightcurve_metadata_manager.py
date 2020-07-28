"""
Code for managing the TESS FFI metadata SQL table.
"""
import itertools
from pathlib import Path
from typing import List, Union, Generator
from peewee import IntegerField, CharField, FloatField, SchemaManager

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase, metadatabase_uuid, \
    convert_class_to_table_name, dataset_split_from_uuid
from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface


class TessFfiLightcurveMetadata(MetadatabaseModel):
    """
    A model for the TESS FFI lightcurve metadatabase table.
    """
    tic_id = IntegerField(index=True)
    sector = IntegerField(index=True)
    path = CharField(unique=True)
    dataset_split = IntegerField()
    magnitude = FloatField()
    random_order_uuid = CharField(unique=True, index=True)

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('tic_id', 'sector'), True),  # Ensures TIC ID and sector entry is unique.
            (('dataset_split', 'random_order_uuid'), False),  # Useful for training data split with random order.
            (('magnitude', 'dataset_split', 'random_order_uuid'), False),  # Common training selection order.
            (('dataset_split', 'tic_id', 'random_order_uuid'), False),  # Common training selection order.
            (('magnitude', 'dataset_split', 'tic_id', 'random_order_uuid'), False)  # Common training selection order.
        )


class TessFfiLightcurveMetadataManager:
    """
    A class for managing the TESS FFI metadata SQL table.
    """
    tess_ffi_data_interface = TessFfiDataInterface()

    def __init__(self):
        self.lightcurve_root_directory_path = Path('data/tess_ffi_lightcurves')

    def insert_multiple_rows_from_paths_into_database(self, lightcurve_paths: List[Path]):
        """
        Inserts sets of lightcurve paths into the table.

        :param lightcurve_paths: The list of paths to insert.
        """
        row_dictionary_list = []
        table_name = convert_class_to_table_name(TessFfiLightcurveMetadata)
        for lightcurve_path in lightcurve_paths:
            tic_id, sector = self.tess_ffi_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
            magnitude = self.tess_ffi_data_interface.get_floor_magnitude_from_file_path(lightcurve_path)
            relative_path = lightcurve_path.relative_to(self.lightcurve_root_directory_path)
            uuid_name = f'{table_name} TIC {tic_id} sector {sector}'
            uuid = metadatabase_uuid(uuid_name)
            dataset_split = dataset_split_from_uuid(uuid)
            row_dictionary_list.append({TessFfiLightcurveMetadata.path.name: str(relative_path),
                                        TessFfiLightcurveMetadata.tic_id.name: tic_id,
                                        TessFfiLightcurveMetadata.sector.name: sector,
                                        TessFfiLightcurveMetadata.magnitude.name: magnitude,
                                        TessFfiLightcurveMetadata.dataset_split.name: dataset_split,
                                        TessFfiLightcurveMetadata.random_order_uuid.name: str(uuid)})
        with metadatabase.atomic():
            TessFfiLightcurveMetadata.insert_many(row_dictionary_list).execute()

    def populate_sql_database(self):
        """
        Populates the SQL database based on the lightcurve files.
        """
        print('Populating the TESS FFI lightcurve meta data table...')
        path_glob = self.lightcurve_root_directory_path.glob('tesslcs_sector_*_104/tesslcs_tmag_*_*/tesslc_*.pkl')
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
        print(f'TESS FFI lightcurve meta data table populated. {row_count} rows added.')

    def build_table(self):
        """
        Builds the SQL table.
        """
        TessFfiLightcurveMetadata.drop_table()
        TessFfiLightcurveMetadata.create_table()
        SchemaManager(TessFfiLightcurveMetadata).drop_indexes()  # To allow for fast insert.
        self.populate_sql_database()
        SchemaManager(TessFfiLightcurveMetadata).create_indexes()  # Since we dropped them before.

    def create_paths_generator(self, magnitude_range: (Union[float, None], Union[float, None]) = (None, None),
                               dataset_splits: Union[List[int], None] = None) -> Generator[Path, None, None]:
        """
        Creates a generator for all the paths from the SQL table, with optional filters.

        :param magnitude_range: The range of magnitudes to consider.
        :param dataset_splits: The dataset splits to filter on. For splitting training, testing, etc.
        :param repeat: Whether or not the generator should repeat indefinitely.
        :return: The generator.
        """
        page_size = 1000
        query = TessFfiLightcurveMetadata().select().order_by(
            TessFfiLightcurveMetadata.random_order_uuid)
        if magnitude_range[0] is not None and magnitude_range[1] is not None:
            query = query.where(TessFfiLightcurveMetadata.magnitude.between(*magnitude_range))
        elif magnitude_range[0] is not None:
            query = query.where(TessFfiLightcurveMetadata.magnitude > magnitude_range[0])
        elif magnitude_range[1] is not None:
            query = query.where(TessFfiLightcurveMetadata.magnitude < magnitude_range[1])
        if dataset_splits is not None:
            query = query.where(TessFfiLightcurveMetadata.dataset_split.in_(dataset_splits))
        for page_number in itertools.count(start=1, step=1):  # Peewee pages start on 1.
            page = query.paginate(page_number, paginate_by=page_size)
            if len(page) > 0:
                for row in page:
                    yield Path(self.lightcurve_root_directory_path.joinpath(row.path))
            else:
                break


if __name__ == '__main__':
    manager = TessFfiLightcurveMetadataManager()
    manager.build_table()
