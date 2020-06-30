"""
Code to represent a database to train to find exoplanet transits 2 minute cadence TESS data.
Uses known TOI dispositions and injects them into other TESS lightcurves to create positive training samples.
"""
from pathlib import Path
import numpy as np
from typing import Iterable
import requests
from astropy.io import fits

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.photometric_database.tess_synthetic_injected_with_negative_injection_database import \
    TessSyntheticInjectedWithNegativeInjectionDatabase


class ToiDatabase(TessSyntheticInjectedWithNegativeInjectionDatabase):
    """
    A class to represent a database to train to find exoplanet transits 2 minute cadence TESS data.
    Uses known TOI dispositions and injects them into other TESS lightcurves to create positive training samples.
    """

    def __init__(self, data_directory='data/toi_database'):
        super().__init__(data_directory=data_directory)
        self.toi_dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')
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

    def download_exofop_toi_database(self, number_of_negative_lightcurves_to_download=10000):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_.
        """
        # print('Clearing data directory...')
        # self.clear_data_directory()
        print("Downloading ExoFOP TOI disposition CSV...")
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with self.toi_dispositions_path.open('wb') as csv_file:
            csv_file.write(response.content)
        print('Downloading TESS observation list...')
        tess_data_interface = TessDataInterface()
        tess_observations = tess_data_interface.get_all_tess_time_series_observations()
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_observations = tess_data_interface.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        print("Downloading lightcurves which are confirmed or suspected planets in TOI dispositions...")
        tess_toi_data_interface = TessToiDataInterface()
        toi_database.tess_data_interface.download_exofop_toi_lightcurves_to_directory(
            toi_database.synthetic_signal_directory)
        toi_dispositions = tess_toi_data_interface.load_toi_dispositions_in_project_format()
        print("Downloading lightcurves which are not in TOI dispositions and do not have TCEs (not planets)...")
        print(f'Download limited to {number_of_negative_lightcurves_to_download} lightcurves...')
        # noinspection SpellCheckingInspection
        toi_tic_ids = toi_dispositions['TIC ID'].values
        not_toi_observations = single_sector_observations[
            ~single_sector_observations['TIC ID'].isin(toi_tic_ids)  # Don't include even false positives.
        ]
        not_toi_observations = not_toi_observations.sample(frac=1, random_state=0)
        # Shorten product list obtaining.
        not_toi_observations = not_toi_observations.head(number_of_negative_lightcurves_to_download * 2)
        not_toi_data_products = tess_data_interface.get_product_list(not_toi_observations)
        not_toi_data_products = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            not_toi_data_products)
        not_toi_lightcurve_data_products = not_toi_data_products[
            not_toi_data_products['productFilename'].str.endswith('lc.fits')
        ]
        not_toi_data_validation_data_products = not_toi_data_products[
            not_toi_data_products['productFilename'].str.endswith('dvr.xml')
        ]
        tic_ids_with_dv = not_toi_data_validation_data_products['TIC ID'].values
        not_planet_lightcurve_data_products = not_toi_lightcurve_data_products[
            ~not_toi_lightcurve_data_products['TIC ID'].isin(tic_ids_with_dv)  # Remove any lightcurves with TCEs.
        ]
        # Shuffle rows.
        not_planet_lightcurve_data_products = not_planet_lightcurve_data_products.sample(frac=1, random_state=0)
        not_planet_download_manifest = tess_data_interface.download_products(
            not_planet_lightcurve_data_products.head(number_of_negative_lightcurves_to_download),
            data_directory=self.data_directory
        )
        print(f'Verifying and moving lightcurves to {self.lightcurve_directory}...')
        self.lightcurve_directory.mkdir(parents=True, exist_ok=True)
        for file_path_string in not_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            lightcurve_path = self.lightcurve_directory.joinpath(file_path.name)
            try:
                file_path.rename(lightcurve_path)
                hdu_list = fits.open(str(lightcurve_path))
                lightcurve = hdu_list[1].data
                _ = lightcurve['TIME'][0]
            except (OSError, TypeError):
                print(f'{file_path} seems to be corrupt. Re-downloading and replacing.')
                sector = tess_data_interface.get_sector_from_single_sector_obs_id(str(lightcurve_path.stem))
                tic_id = tess_data_interface.get_tic_id_from_single_sector_obs_id(str(lightcurve_path.stem))
                tess_data_interface.download_lightcurve(tic_id, sector, save_directory=lightcurve_path.parent)
        print('Database ready.')


if __name__ == '__main__':
    toi_database = ToiDatabase()
    toi_database.download_exofop_toi_database(number_of_negative_lightcurves_to_download=1000000)
