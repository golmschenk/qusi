"""
Code to represent a TESS light curve.
"""
from typing import Union

from ramjet.photometric_database.light_curve import LightCurve


class TessLightCurve(LightCurve):
    """
    A class to represent a TESS light curve.
    """
    def __init__(self):
        super().__init__()
        self.tic_id: Union[int, None] = None
        self.sector: Union[int, None] = None
