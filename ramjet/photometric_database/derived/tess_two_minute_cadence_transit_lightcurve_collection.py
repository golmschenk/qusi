"""
Code representing the collection of TESS two minute cadence lightcurves containing transits.
"""
import itertools
import numpy as np
from collections import Iterable
from pathlib import Path
from typing import List, Union

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadata, Disposition
from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import \
    TessTwoMinuteCadenceLightcurveMetadata, TessTwoMinuteCadenceLightcurveMetadataManger
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


class TessTwoMinuteCadenceTransitLightcurveCollection(LightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    tess_data_interface = TessDataInterface()
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()

    def __init__(self, dataset_splits: Union[List[int], None] = None, repeat=True):
        super().__init__()
        self.label = 1
        self.dataset_splits: Union[List[int], None] = dataset_splits
        self.repeat = repeat

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        page_size = 1000
        query = TessTwoMinuteCadenceLightcurveMetadata().select().order_by(
            TessTwoMinuteCadenceLightcurveMetadata.random_order_uuid)
        if self.dataset_splits is not None:
            query = query.where(TessTwoMinuteCadenceLightcurveMetadata.dataset_split.in_(self.dataset_splits))
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
        query = query.where(TessTwoMinuteCadenceLightcurveMetadata.tic_id.in_(transit_tic_id_query))
        while True:
            for page_number in itertools.count(start=1, step=1):  # Peewee pages start on 1.
                page = query.paginate(page_number, paginate_by=page_size)
                if len(page) > 0:
                    for row in page:
                        yield Path(self.tess_two_minute_cadence_lightcurve_metadata_manger.
                                   lightcurve_root_directory_path.joinpath(row.path))
                else:
                    break
            if not self.repeat:
                break

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path)
        return times, fluxes


class TessTwoMinuteCadenceNonTransitLightcurveCollection(LightcurveCollection):
    """
    A class representing the collection of TESS two minute cadence lightcurves containing transits.
    """
    tess_data_interface = TessDataInterface()
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()

    def __init__(self, dataset_splits: Union[List[int], None] = None, repeat=True):
        super().__init__()
        self.label = 0
        self.dataset_splits: Union[List[int], None] = dataset_splits
        self.repeat = repeat

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        page_size = 1000
        query = TessTwoMinuteCadenceLightcurveMetadata().select().order_by(
            TessTwoMinuteCadenceLightcurveMetadata.random_order_uuid)
        if self.dataset_splits is not None:
            query = query.where(TessTwoMinuteCadenceLightcurveMetadata.dataset_split.in_(self.dataset_splits))
        transit_candidate_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value |
            TessTransitMetadata.disposition == Disposition.CANDIDATE.value)
        query = query.where(TessTwoMinuteCadenceLightcurveMetadata.tic_id.not_in(transit_candidate_tic_id_query))
        while True:
            for page_number in itertools.count(start=1, step=1):  # Peewee pages start on 1.
                page = query.paginate(page_number, paginate_by=page_size)
                if len(page) > 0:
                    for row in page:
                        yield Path(self.tess_two_minute_cadence_lightcurve_metadata_manger.
                                   lightcurve_root_directory_path.joinpath(row.path))
                else:
                    break
            if not self.repeat:
                break

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given lightcurve path.

        :param path: The path to the lightcurve file.
        :return: The times and the fluxes of the lightcurve.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path)
        return times, fluxes
