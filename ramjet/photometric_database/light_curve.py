"""
Code for a class to represent a light curve. See the contained class docstring for more details.
"""
from typing import Union, Dict

import numpy as np


class LightCurve:
    """
    A class to represent a light curve. A light curve is a collection of data which includes times and fluxes
    (often several types of fluxes).
    """
    def __init__(self):
        self.times: np.ndarray
        self.fluxes_dictionary: Dict[np.ndarray] = {}
        self.default_flux_type: Union[str, None] = None

    @property
    def fluxes(self) -> np.ndarray:
        """
        The fluxes of the lightcurve. Uses the default flux type if multiple are available.

        :return: The fluxes.
        """
        number_of_flux_types = len(self.fluxes_dictionary)
        if number_of_flux_types == 1:
            return next(iter(self.fluxes_dictionary.values()))
        elif number_of_flux_types > 1:
            if self.default_flux_type is None:
                raise ValueError('A light curve with multiple flux types must have a default type to use `fluxes`.')
            else:
                return self.fluxes_dictionary[self.default_flux_type]
        else:
            raise ValueError('The lightcurve contains no fluxes.')
