"""
Code to represent a database to train to find exoplanet transits 2 minute cadence TESS data.
Uses known TOI dispositions and injects them into other TESS lightcurves to create positive training samples.
"""
import numpy as np
from typing import List
from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase


class ToiDatabase(TessSyntheticInjectedDatabase):
    """
    A class to represent a database to train to find exoplanet transits 2 minute cadence TESS data.
    Uses known TOI dispositions and injects them into other TESS lightcurves to create positive training samples.
    """

    def __init__(self, data_directory='data/toi_database'):
        super().__init__(data_directory=data_directory)
        self.toi_dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')
        self.allow_out_of_bounds_injection = True

    def get_all_synthetic_signal_paths(self) -> List[str]:
        """
        Returns the list of all synthetic signals to use.

        :return: The list of synthetic signals.
        """
        synthetic_signal_paths = list(map(str, self.synthetic_signal_directory.glob('**/*.fits')))
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


if __name__ == '__main__':
    toi_database = ToiDatabase()
    toi_database.tess_data_interface.download_exofop_toi_lightcurves_to_directory(
        toi_database.synthetic_signal_directory)
    toi_database.tess_data_interface.download_all_two_minute_cadence_lightcurves(toi_database.lightcurve_directory)
