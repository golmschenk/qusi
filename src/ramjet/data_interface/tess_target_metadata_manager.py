"""
Code for managing the metadata of the TESS targets.
"""
import logging
from pathlib import Path

from peewee import IntegerField

from ramjet.data_interface.metadatabase import (
    MetadatabaseModel,
    convert_class_to_table_name,
    dataset_split_from_uuid,
    metadatabase,
    metadatabase_uuid,
)
from ramjet.data_interface.tess_data_interface import get_tic_id_and_sector_from_file_path

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.light_curve_root_directory_path = Path("data/tess_two_minute_cadence_light_curves")

    def insert_multiple_rows_from_paths_into_database(self, light_curve_paths: list[Path]) -> int:
        """
        Inserts sets targets into the table from light curve paths.

        :param light_curve_paths: The list of paths to insert.
        :return: The number of rows inserted.
        """
        row_dictionary_list = []
        table_name = convert_class_to_table_name(TessTargetMetadata)
        for light_curve_path in light_curve_paths:
            tic_id, _ = get_tic_id_and_sector_from_file_path(light_curve_path)
            uuid_name = f"{table_name} TIC {tic_id}"
            uuid = metadatabase_uuid(uuid_name)
            dataset_split = dataset_split_from_uuid(uuid)
            row_dictionary_list.append(
                {TessTargetMetadata.tic_id.name: tic_id, TessTargetMetadata.dataset_split.name: dataset_split}
            )
        with metadatabase.atomic():
            number_of_rows_inserted = TessTargetMetadata.insert_many(row_dictionary_list).on_conflict_ignore().execute()
        return number_of_rows_inserted

    def populate_sql_database(self):
        """
        Populates the SQL database based on the light curve files.
        """
        logger.info("Populating the TESS target light curve metadata table...")
        path_glob = self.light_curve_root_directory_path.glob("**/*.fits")
        row_count = 0
        batch_paths = []
        batch_dataset_splits = []
        with metadatabase.atomic():
            for index, path in enumerate(path_glob):
                batch_paths.append(path.relative_to(self.light_curve_root_directory_path))
                batch_dataset_splits.append(index % 10)
                if index % 1000 == 0 and index != 0:
                    row_count += self.insert_multiple_rows_from_paths_into_database(batch_paths)
                    batch_paths = []
                    batch_dataset_splits = []
                    logger.info(f"{row_count} rows inserted...")
            if len(batch_paths) > 0:
                row_count += self.insert_multiple_rows_from_paths_into_database(batch_paths)
        logger.info(f"TESS target metadata table populated. {row_count} rows added.")

    def build_table(self):
        """
        Builds the SQL table.
        """
        TessTargetMetadata.drop_table()
        TessTargetMetadata.create_table()
        self.populate_sql_database()


if __name__ == "__main__":
    manager = TessTargetMetadataManger()
    manager.build_table()
