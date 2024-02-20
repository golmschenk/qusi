"""
Code for a light curve collection of the TESS two minute cadence data.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ramjet.data_interface.tess_data_interface import (
    TessDataInterface,
    TessFluxType,
    download_two_minute_cadence_light_curves,
    load_fluxes_and_times_from_fits_file,
)
from ramjet.data_interface.tess_target_metadata_manager import TessTargetMetadata
from ramjet.data_interface.tess_two_minute_cadence_light_curve_metadata_manager import (
    TessTwoMinuteCadenceLightCurveMetadata,
    TessTwoMinuteCadenceLightCurveMetadataManger,
)
from ramjet.photometric_database.sql_metadata_light_curve_collection import SqlMetadataLightCurveCollection

if TYPE_CHECKING:
    import numpy as np
    from peewee import Select

    from ramjet.data_interface.metadatabase import MetadatabaseModel


class TessTwoMinuteCadenceLightCurveCollection(SqlMetadataLightCurveCollection):
    """
    A light curve collection of the TESS two minute cadence data.
    """

    tess_data_interface = TessDataInterface()
    tess_two_minute_cadence_light_curve_metadata_manger = TessTwoMinuteCadenceLightCurveMetadataManger()

    def __init__(self, dataset_splits: list[int] | None = None, flux_type: TessFluxType = TessFluxType.PDCSAP):
        super().__init__()
        self.data_directory: Path = Path("data/tess_two_minute_cadence_light_curves")
        self.label = 0
        self.dataset_splits: list[int] | None = dataset_splits
        self.flux_type: TessFluxType = flux_type

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = TessTwoMinuteCadenceLightCurveMetadata().select(TessTwoMinuteCadenceLightCurveMetadata.path)
        if self.dataset_splits is not None:
            query = query.where(TessTwoMinuteCadenceLightCurveMetadata.dataset_split.in_(self.dataset_splits))
        return query

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the light curve path from the SQL database model.

        :return: The path to the light curve.
        """
        return Path(
            self.tess_two_minute_cadence_light_curve_metadata_manger.light_curve_root_directory_path.joinpath(
                model.path
            )
        )

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        fluxes, times = load_fluxes_and_times_from_fits_file(path, self.flux_type)
        return times, fluxes

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a given path as an injectable signal.

        :param path: The path to the light curve/signal file.
        :return: The times and the magnifications of the light curve/signal.
        """
        fluxes, times = load_fluxes_and_times_from_fits_file(path, self.flux_type)
        magnifications, times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return times, magnifications

    def download(self):
        """
        Downloads the light curve collection.
        """
        download_two_minute_cadence_light_curves(self.data_directory)


class TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection(TessTwoMinuteCadenceLightCurveCollection):
    """
    A light curve collection of the TESS two minute cadence data with light curves from the same target in the same
    dataset split.
    """

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = TessTwoMinuteCadenceLightCurveMetadata().select(TessTwoMinuteCadenceLightCurveMetadata.path)
        query = query.join(
            TessTargetMetadata, on=TessTwoMinuteCadenceLightCurveMetadata.tic_id == TessTargetMetadata.tic_id
        )
        if self.dataset_splits is not None:
            query = query.where(TessTargetMetadata.dataset_split.in_(self.dataset_splits))
        return query
