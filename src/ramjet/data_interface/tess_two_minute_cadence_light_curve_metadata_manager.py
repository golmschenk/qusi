"""
Code for managing the meta data of the two minute cadence TESS light curves.
"""
import logging
from pathlib import Path

from peewee import CharField, IntegerField, SchemaManager

from ramjet.data_interface.metadatabase import (
    MetadatabaseModel,
    convert_class_to_table_name,
    dataset_split_from_uuid,
    metadatabase,
    metadatabase_uuid,
)
from ramjet.data_interface.tess_data_interface import get_tic_id_and_sector_from_file_path

logger = logging.getLogger(__name__)


class TessTwoMinuteCadenceLightCurveMetadata(MetadatabaseModel):
    """
    A model for the TESS two minute cadence light curve metadatabase table.
    """

    tic_id = IntegerField(index=True)
    sector = IntegerField(index=True)
    path = CharField(unique=True)
    dataset_split = IntegerField()

    class Meta:
        """Schema meta data for the model."""

        indexes = (
            (("sector", "tic_id"), True),  # Ensures TIC ID and sector entry is unique.
        )


class TessTwoMinuteCadenceLightCurveMetadataManger:
    """
    A class for managing the metadata of the two-minute cadence TESS light curves.
    """

    def __init__(self):
        self.light_curve_root_directory_path = Path("data/tess_two_minute_cadence_light_curves")

    def insert_multiple_rows_from_paths_into_database(self, light_curve_paths: list[Path]):
        """
        Inserts sets of light curve paths into the table.

        :param light_curve_paths: The list of paths to insert.
        """
        row_dictionary_list = []
        table_name = convert_class_to_table_name(TessTwoMinuteCadenceLightCurveMetadata)
        for light_curve_path in light_curve_paths:
            tic_id, sector = get_tic_id_and_sector_from_file_path(light_curve_path)
            uuid_name = f"{table_name} TIC {tic_id} sector {sector}"
            uuid = metadatabase_uuid(uuid_name)
            dataset_split = dataset_split_from_uuid(uuid)
            row_dictionary_list.append(
                {
                    TessTwoMinuteCadenceLightCurveMetadata.path.name: str(light_curve_path),
                    TessTwoMinuteCadenceLightCurveMetadata.tic_id.name: tic_id,
                    TessTwoMinuteCadenceLightCurveMetadata.sector.name: sector,
                    TessTwoMinuteCadenceLightCurveMetadata.dataset_split.name: dataset_split,
                }
            )
        with metadatabase.atomic():
            TessTwoMinuteCadenceLightCurveMetadata.insert_many(row_dictionary_list).execute()

    def populate_sql_database(self):
        """
        Populates the SQL database based on the light curve files.
        """
        logger.info("Populating the TESS two minute cadence light curve meta data table...")
        path_glob = self.light_curve_root_directory_path.glob("**/*.fits")
        row_count = 0
        batch_paths = []
        with metadatabase.atomic():
            for index, path in enumerate(path_glob):
                batch_paths.append(path.relative_to(self.light_curve_root_directory_path))
                row_count += 1
                if index % 1000 == 0 and index != 0:
                    self.insert_multiple_rows_from_paths_into_database(batch_paths)
                    batch_paths = []
                    logger.info(f"{index} rows inserted...")
            if len(batch_paths) > 0:
                self.insert_multiple_rows_from_paths_into_database(batch_paths)
        logger.info(f"TESS two minute cadence light curve meta data table populated. {row_count} rows added.")

    def build_table(self):
        """
        Builds the SQL table.
        """
        TessTwoMinuteCadenceLightCurveMetadata.drop_table()
        TessTwoMinuteCadenceLightCurveMetadata.create_table()
        SchemaManager(TessTwoMinuteCadenceLightCurveMetadata).drop_indexes()  # To allow for fast insert.
        self.populate_sql_database()
        logger.info("Building indexes...")
        SchemaManager(TessTwoMinuteCadenceLightCurveMetadata).create_indexes()  # Since we dropped them before.


if __name__ == "__main__":
    manager = TessTwoMinuteCadenceLightCurveMetadataManger()
    manager.build_table()
