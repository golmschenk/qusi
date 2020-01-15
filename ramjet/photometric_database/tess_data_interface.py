"""
Code for a class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
"""
from astroquery.mast import Observations


class TessDataInterface:
    """
    A class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
    """
    def __init__(self):
        Observations.TIMEOUT = 1200
        Observations.PAGESIZE = 10000
