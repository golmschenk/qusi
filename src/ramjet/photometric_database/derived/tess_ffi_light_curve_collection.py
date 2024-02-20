"""
Code for a light curve collection of the TESS FFI data, as produced by Brian Powell.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ramjet.data_interface.tess_ffi_light_curve_metadata_manager import (
    TessFfiLightCurveMetadata,
    TessFfiLightCurveMetadataManager,
)
from ramjet.photometric_database.sql_metadata_light_curve_collection import SqlMetadataLightCurveCollection
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve

if TYPE_CHECKING:
    import numpy as np
    from peewee import Select

    from ramjet.data_interface.metadatabase import MetadatabaseModel


class TessFfiLightCurveCollection(SqlMetadataLightCurveCollection):
    """
    A light curve collection of the TESS two minute cadence data.
    """

    tess_ffi_light_curve_metadata_manger = TessFfiLightCurveMetadataManager()

    def __init__(
        self, dataset_splits: list[int] | None = None, magnitude_range: (float | None, float | None) = (None, None)
    ):
        super().__init__()
        self.data_directory: Path = Path("data/tess_ffi_light_curves")
        self.label = 0
        self.dataset_splits: list[int] | None = dataset_splits
        self.magnitude_range: (float | None, float | None) = magnitude_range

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = TessFfiLightCurveMetadata().select()
        if self.magnitude_range[0] is not None and self.magnitude_range[1] is not None:
            query = query.where(TessFfiLightCurveMetadata.magnitude.between(*self.magnitude_range))
        elif self.magnitude_range[0] is not None:
            query = query.where(TessFfiLightCurveMetadata.magnitude > self.magnitude_range[0])
        elif self.magnitude_range[1] is not None:
            query = query.where(TessFfiLightCurveMetadata.magnitude < self.magnitude_range[1])
        if self.dataset_splits is not None:
            query = query.where(TessFfiLightCurveMetadata.dataset_split.in_(self.dataset_splits))
        return query

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the light curve path from the SQL database model.

        :return: The path to the light curve.
        """
        return Path(self.tess_ffi_light_curve_metadata_manger.light_curve_root_directory_path.joinpath(model.path))

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        fluxes, times = TessFfiLightCurve.load_fluxes_and_times_from_pickle_file(path)
        return times, fluxes

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the light curve/signal file.
        :return: The times and the magnifications of the light curve/signal.
        """
        fluxes, times = TessFfiLightCurve.load_fluxes_and_times_from_pickle_file(path)
        magnifications, times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return times, magnifications
