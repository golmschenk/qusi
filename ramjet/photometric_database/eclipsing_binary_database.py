"""
Code to search for eclipsing binaries.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.photometric_database.tess_synthetic_injected_with_negative_injection_database import \
    TessSyntheticInjectedWithNegativeInjectionDatabase


class EclipsingBinaryDatabase(TessSyntheticInjectedWithNegativeInjectionDatabase):
    """
    A class to represent a database to train to find eclipsing binaries in 2 minute cadence TESS data.
    Uses known cases from Brian Powell's eclipsing binary catalog.
    """

    def __init__(self, data_directory='data/eclipsing_binary_database'):
        super().__init__(data_directory=data_directory)
        self.catalog_csv_path = self.data_directory.joinpath('ebcat_partial_sectors.csv')
        self.allow_out_of_bounds_injection = True

    def get_all_synthetic_signal_paths(self) -> Iterable[Path]:
        """
        Returns the list of all synthetic signals to use.

        :return: The list of synthetic signals.
        """
        synthetic_signal_paths = self.synthetic_signal_directory.glob('**/*.fits')
        return synthetic_signal_paths

    def load_magnifications_and_times_from_synthetic_signal_path(self, synthetic_signal_path: str
                                                                 ) -> (np.ndarray, np.ndarray):
        """
        Loads the synthetic signal from the path given.

        :param synthetic_signal_path: The path to the synthetic signal data file.
        :return: The magnifications and relative times of the synthetic signal.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(synthetic_signal_path)
        synthetic_magnifications, synthetic_times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return synthetic_magnifications, synthetic_times

    def download_catalog_eclipsing_binaries(self):
        """
        Downloads the eclipsing binaries listed in Brian Powell's catalog to the synthetic signals directory.
        """
        catalog = pd.read_csv(self.catalog_csv_path)
        catalog = catalog[catalog['2min'] == 1]
        tess_data_interface = TessDataInterface()
        tess_observations = tess_data_interface.get_all_tess_time_series_observations(tic_id=catalog['ID'])
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_observations = tess_data_interface.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_data_products = tess_data_interface.get_product_list(single_sector_observations)
        data_products = single_sector_data_products[
            single_sector_data_products['productFilename'].str.endswith('lc.fits')
        ]
        download_manifest = self.tess_data_interface.download_products(
            data_products, data_directory=self.data_directory)
        print(f'Moving lightcurves to {self.synthetic_signal_directory}...')
        self.synthetic_signal_directory.mkdir(parents=True, exist_ok=True)
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.synthetic_signal_directory.joinpath(file_path.name))


if __name__ == '__main__':
    database = EclipsingBinaryDatabase()
    database.download_catalog_eclipsing_binaries()
