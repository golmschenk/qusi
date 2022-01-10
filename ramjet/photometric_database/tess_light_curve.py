"""
Code to represent a TESS light curve.
"""
from typing import Union

from astropy import units
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

from ramjet.photometric_database.light_curve import LightCurve


class TessLightCurve(LightCurve):
    """
    A class to represent a TESS light curve.
    """
    def __init__(self):
        super().__init__()
        self.tic_id: Union[int, None] = None
        self.sector: Union[int, None] = None

    @property
    def sky_coord(self) -> SkyCoord:
        tic_row = Catalogs.query_object(f'TIC{self.tic_id}', catalog='TIC').to_pandas().iloc[0]
        sky_coord = SkyCoord(ra=tic_row['ra'], dec=tic_row['dec'], unit=units.deg)
        return sky_coord

    @property
    def tess_magnitude(self) -> float:
        tic_row = Catalogs.query_object(f'TIC{self.tic_id}', catalog='TIC').to_pandas().iloc[0]
        return float(tic_row['Tmag'])
