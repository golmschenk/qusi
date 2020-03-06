"""
Code to represent a database to train to find exoplanet transits in FFI data based on known TOI dispositions.
"""
import pickle
import requests
import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, List
from pathlib import Path

from ramjet.photometric_database.tess_data_interface import TessDataInterface
from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase


class FfiDataIndexes(Enum):
    """
    An enum for accessing Brian Powell's FFI pickle data with understandable indexes.
    """
    TIC_ID = 0
    RA = 1
    DEC = 2
    TESS_MAGNITUDE = 3
    TIME = 4
    RAW_FLUX = 5
    CORRECTED_FLUX = 6
    PCA_FLUX = 7
    FLUX_ERROR = 8


class ToiColumns(Enum):
    """
    An enum for the names of the TOI columns for Pandas data frames.
    """
    tic_id = 'TIC ID'
    disposition = 'Disposition'
    planet_number = 'Planet number'
    transit_epoch__bjd = 'Transit epoch (BJD)'
    transit_period__days = 'Transit period (days)'
    transit_duration = 'Transit duration (hours)'
    sector = 'Sector'


class FfiToiDatabase(TessSyntheticInjectedDatabase):
    """
    Code to represent a database to train to find exoplanet transits in FFI data based on known TOI dispositions.
    """
    def __init__(self, data_directory='data/tess_toi_ffi'):
        super().__init__(data_directory=data_directory)
        self.toi_dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')
        self.time_steps_per_example = 1296  # 27 days / 30 minutes.

    @staticmethod
    def load_fluxes_and_times_from_ffi_pickle_file(file_path: Union[Path, str]) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with file_path.open('rb') as pickle_file:
            lightcurve = pickle.load(pickle_file)
        fluxes = lightcurve[FfiDataIndexes.CORRECTED_FLUX.value]
        times = lightcurve[FfiDataIndexes.TIME.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    @staticmethod
    def generate_synthetic_signal_from_real_data(fluxes: np.ndarray, times: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Takes real lightcurve data and converts it to a form that can be used for synthetic lightcurve injection.

        :param fluxes: The real lightcurve fluxes.
        :param times: The real lightcurve times.
        :return: Fake synthetic magnifications and times.
        """
        flux_median = np.median(fluxes)
        normalized_fluxes = fluxes / flux_median
        relative_times = times - np.min(times)
        return normalized_fluxes, relative_times

    def download_exofop_toi_lightcurves_to_synthetic_directory(self):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_ lightcurve files to the
        synthetic directory.
        """
        print("Downloading ExoFOP TOI disposition CSV...")
        self.create_data_directories()
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with self.toi_dispositions_path.open('wb') as csv_file:
            csv_file.write(response.content)
        toi_dispositions = self.load_toi_dispositions_in_project_format()
        tic_ids = toi_dispositions[ToiColumns.tic_id.value].unique()
        print('Downloading TESS obdservation list...')
        tess_data_interface = TessDataInterface()
        tess_observations = tess_data_interface.get_all_tess_time_series_observations(tic_id=tic_ids)
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_observations = tess_data_interface.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        print("Downloading lightcurves which are confirmed or suspected planets in TOI dispositions...")
        suspected_planet_dispositions = toi_dispositions[toi_dispositions[ToiColumns.disposition.value] != 'FP']
        suspected_planet_observations = pd.merge(single_sector_observations, suspected_planet_dispositions, how='inner',
                                                 on=[ToiColumns.tic_id.value, ToiColumns.sector.value])
        observations_not_found = suspected_planet_dispositions.shape[0] - suspected_planet_observations.shape[0]
        print(f"{suspected_planet_observations.shape[0]} observations found that match the TOI dispositions.")
        print(f"No observations found for {observations_not_found} entries in TOI dispositions.")
        suspected_planet_data_products = tess_data_interface.get_product_list(suspected_planet_observations)
        suspected_planet_lightcurve_data_products = suspected_planet_data_products[
            suspected_planet_data_products['productFilename'].str.endswith('lc.fits')
        ]
        suspected_planet_download_manifest = tess_data_interface.download_products(
            suspected_planet_lightcurve_data_products, data_directory=self.data_directory)
        print(f'Moving lightcurves to {self.synthetic_signal_directory}...')
        for file_path_string in suspected_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.synthetic_signal_directory.joinpath(file_path.name))

    def load_toi_dispositions_in_project_format(self) -> pd.DataFrame:
        """
        Loads the ExoFOP TOI table information from CSV to a data frame using a project consistent naming scheme.

        :return: The data frame of the TOI dispositions table.
        """
        columns_to_use = ['TIC ID', 'TFOPWG Disposition', 'Planet Num', 'Epoch (BJD)', 'Period (days)',
                          'Duration (hours)', 'Sectors']
        dispositions = pd.read_csv(self.toi_dispositions_path, usecols=columns_to_use)
        dispositions.rename(columns={'TFOPWG Disposition': ToiColumns.disposition.value,
                                     'Planet Num': ToiColumns.planet_number.value,
                                     'Epoch (BJD)': ToiColumns.transit_epoch__bjd.value,
                                     'Period (days)': ToiColumns.transit_period__days.value,
                                     'Duration (hours)': ToiColumns.transit_duration.value,
                                     'Sectors': ToiColumns.sector.value}, inplace=True)
        dispositions[ToiColumns.disposition.value] = dispositions[ToiColumns.disposition.value].fillna('')
        dispositions = dispositions[dispositions[ToiColumns.sector.value].notna()]
        dispositions[ToiColumns.sector.value] = dispositions[ToiColumns.sector.value].str.split(',')
        dispositions = dispositions.explode(ToiColumns.sector.value)
        dispositions[ToiColumns.sector.value] = pd.to_numeric(dispositions[ToiColumns.sector.value]
                                                              ).astype(pd.Int64Dtype())
        return dispositions

    def create_data_directories(self):
        """
        Creates the data directories to be used by the database.
        """
        super().create_data_directories()
        self.lightcurve_directory.mkdir(parents=True, exist_ok=True)
        self.synthetic_signal_directory.mkdir(parents=True, exist_ok=True)

    def get_all_lightcurve_paths(self) -> List[str]:
        """
        Returns the list of all lightcurves to use.

        :return: The list of lightcurves.
        """
        lightcurve_paths = list(map(str, self.lightcurve_directory.glob('**/*.pkl')))
        return lightcurve_paths

    def get_all_synthetic_signal_paths(self) -> List[str]:
        """
        Returns the list of all synthetic signals to use.

        :return: The list of synthetic signals.
        """
        synthetic_signal_paths = list(map(str, self.synthetic_signal_directory.glob('**/*.fits')))
        return synthetic_signal_paths

    def load_fluxes_and_times_from_lightcurve_path(self, lightcurve_path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the lightcurve from the path given. Should be overridden to fit a specific database's file format.

        :param lightcurve_path: The path to the lightcurve file.
        :return: The fluxes and times of the lightcurve
        """
        fluxes, times = self.load_fluxes_and_times_from_ffi_pickle_file(lightcurve_path)
        nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
        fluxes = np.delete(fluxes, nan_indexes)
        times = np.delete(times, nan_indexes)
        return fluxes, times

    def load_magnifications_and_times_from_synthetic_signal_path(self, synthetic_signal_path: str
                                                                 ) -> (np.ndarray, np.ndarray):
        """
        Loads the synthetic signal from the path given. Should be overridden to fit a specific database's file format.

        :param synthetic_signal_path: The path to the synthetic signal data file.
        :return: The magnifications and relative times of the synthetic signal.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return synthetic_magnifications, synthetic_times


if __name__ == '__main__':
    ffi_toi_database = FfiToiDatabase()
    ffi_toi_database.download_exofop_toi_lightcurves_to_synthetic_directory()
