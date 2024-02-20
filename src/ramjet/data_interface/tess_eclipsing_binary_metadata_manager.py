"""
Code for managing the TESS eclipsing binary metadata.
"""
import logging
from pathlib import Path

import pandas as pd
from peewee import IntegerField, SchemaManager

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase

brian_powell_eclipsing_binary_csv_path = Path("data/tess_eclipsing_binaries/TESS_EB_catalog_23Jun.csv")

logger = logging.getLogger(__name__)


class TessEclipsingBinaryMetadata(MetadatabaseModel):
    """
    A model for the TESS eclipsing binary metadatabase table.
    """

    tic_id = IntegerField(index=True, unique=True)


class TessEclipsingBinaryMetadataManager:
    """
    A class for managing the TESS eclipsing binary metadata.
    """

    @staticmethod
    def build_table():
        """
        Builds the TESS eclipsing binary metadata table.
        """
        logger.info("Building TESS eclipsing binary metadata table...")
        eclipsing_binary_data_frame = pd.read_csv(brian_powell_eclipsing_binary_csv_path, usecols=["ID"])
        row_count = 0
        metadatabase.drop_tables([TessEclipsingBinaryMetadata])
        metadatabase.create_tables([TessEclipsingBinaryMetadata])
        SchemaManager(TessEclipsingBinaryMetadata).drop_indexes()
        rows = []
        for tic_id in eclipsing_binary_data_frame["ID"].values:
            row = {"tic_id": tic_id}
            rows.append(row)
            row_count += 1
            if row_count % 1000 == 0:
                with metadatabase.atomic():
                    TessEclipsingBinaryMetadata.insert_many(rows).execute()
                rows = []
        with metadatabase.atomic():
            TessEclipsingBinaryMetadata.insert_many(rows).execute()
        SchemaManager(TessEclipsingBinaryMetadata).create_indexes()
        logger.info(f"Table built. {row_count} rows added.")


if __name__ == "__main__":
    metadata_manager = TessEclipsingBinaryMetadataManager()
    metadata_manager.build_table()
