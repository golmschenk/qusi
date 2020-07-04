"""
Code for managing the TESS FFI metadata SQL table.
"""
from pathlib import Path
from typing import List
from uuid import uuid4
from peewee import IntegerField, CharField, FloatField

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase
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
    random_order_uuid = CharField(unique=True, index=True, default=uuid4)

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('tic_id', 'sector'), True),  # Ensures TIC ID and sector entry is unique.
            (('dataset_split', 'random_order_uuid'), False),  # Useful for training data split with random order.
            (('magnitude', 'dataset_split', 'random_order_uuid'), False)  # Common training selection order.
        )


class TessFfiLightcurveMetadataManager:
    """
    A class for managing the TESS FFI metadata SQL table.
    """
    tess_ffi_data_interface = TessFfiDataInterface()

    def insert_multiple_rows_from_paths_into_database(self, lightcurve_paths: List[Path], dataset_splits: List[int]):
        """
        Inserts sets of lightcurve paths into the table.

        :param lightcurve_paths: The list of paths to insert.
        :param dataset_splits: The dataset splits to assign to each path.
        """
        assert len(lightcurve_paths) == len(dataset_splits)
        row_dictionary_list = []
        for lightcurve_path, dataset_split in zip(lightcurve_paths, dataset_splits):
            tic_id, sector = self.tess_ffi_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
            magnitude = self.tess_ffi_data_interface.get_magnitude_from_file(lightcurve_path)
            row_dictionary_list.append({TessFfiLightcurveMetadata.path.name: str(lightcurve_path),
                                        TessFfiLightcurveMetadata.tic_id.name: tic_id,
                                        TessFfiLightcurveMetadata.sector.name: sector,
                                        TessFfiLightcurveMetadata.magnitude.name: magnitude,
                                        TessFfiLightcurveMetadata.dataset_split.name: dataset_split})
        with metadatabase.atomic():
            TessFfiLightcurveMetadata.insert_many(row_dictionary_list).execute()
