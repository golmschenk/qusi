"""
Code for a signal to noise calculator for LISA.
"""
from pathlib import Path

import numpy as np
import pandas as pd


class LisaSignalToNoiseCalculator:
    """
    A signal to noise calculator for LISA, based on John Baker's quick formula.
    Uses hard coded reference values calculated by John Baker.
    """
    reference_file_path = Path(__file__).parent.joinpath('lisa_reference_signal_to_noise_ratios.feather')
    reference_data_frame = pd.read_feather(reference_file_path).sort_values(by='Orbital period (seconds)')

    def calculate_median_signal_to_noise_ratio(self, orbital_period: float, distance: float,
                                               mass0: float, mass1: float):
        """
        Calculates the median SNR for a given binary based on John Baker's formula and reference values.

        :param orbital_period: The orbital period in days of the binary.
        :param distance: The distance to the target.
        :param mass0: The mass of the first star.
        :param mass1: The mass of the second star.
        :return: The median LISA SNR estimate.
        """
        orbital_period__seconds = orbital_period * 86400
        interpreted_ratios = np.interp(orbital_period__seconds, self.reference_data_frame['Orbital period (seconds)'],
                                       self.reference_data_frame['Median signal to noise ratio'])
        total_mass = mass0 + mass1
        snr = 4 * mass0 * mass1 / total_mass ** 2 * (total_mass / 2) ** (5 / 3) * (100 / distance) * interpreted_ratios
        return snr
