"""
Code to represent a TESS light curve.
"""
from typing import Union, Optional

import pandas as pd
from astropy import units
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from retrying import retry

from ramjet.data_interface.tess_data_interface import is_common_mast_connection_error
from ramjet.photometric_database.light_curve import LightCurve


class TessLightCurve(LightCurve):
    """
    A class to represent a TESS light curve.
    """
    def __init__(self):
        super().__init__()
        self.tic_id: Union[int, None] = None
        self.sector: Union[int, None] = None
        self._tic_row: Optional[pd.Series] = None

    @property
    @retry(retry_on_exception=is_common_mast_connection_error)
    def sky_coord(self) -> SkyCoord:
        tic_row = self.get_tic_row()
        sky_coord = SkyCoord(ra=tic_row['ra'], dec=tic_row['dec'], unit=units.deg)
        return sky_coord

    def get_tic_row(self):
        if self._tic_row is None:
            self._tic_row = Catalogs.query_object(f'TIC{self.tic_id}', catalog='TIC').to_pandas().iloc[0]
        return self._tic_row

    @property
    @retry(retry_on_exception=is_common_mast_connection_error)
    def tess_magnitude(self) -> float:
        tic_row = self.get_tic_row()
        return float(tic_row['Tmag'])
