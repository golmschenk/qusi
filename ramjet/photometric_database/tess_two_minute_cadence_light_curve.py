"""
Code to for a class to represent a TESS two minute cadence light curve based on a file.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Union
from astropy.io import fits

from ramjet.photometric_database.light_curve import LightCurve


class TessTwoMinuteCadenceColumnName(Enum):
    """
    An enum to represent the column names of the TESS two minute cadence data.
    """
    TIME = 'time__btjd'
    SAP_FLUX = 'sap_flux'
    PDCSAP_FLUX = 'pdcsap_flux'
    SAP_FLUX_ERROR = 'sap_flux_error'
    PDCSAP_FLUX_ERROR = 'pdcsap_flux_error'


class TessTwoMinuteCadenceMastFitsIndex(Enum):
    """
    An enum to represent the indexes of the TESS two minute cadence data in MAST FITS files.
    """
    TIME = 'TIME'
    SAP_FLUX = 'SAP_FLUX'
    PDCSAP_FLUX = 'PDCSAP_FLUX'
    SAP_FLUX_ERROR = 'SAP_FLUX_ERR'
    PDCSAP_FLUX_ERROR = 'PDCSAP_FLUX_ERR'


class TessTwoMinuteCadenceFileBasedLightCurve(LightCurve):
    """
    A class to represent a TESS two minute cadence light curve.
    """
    @classmethod
    def from_path(cls, path: Path,
                  flux_column_name: TessTwoMinuteCadenceColumnName = TessTwoMinuteCadenceColumnName.PDCSAP_FLUX.value,
                  fits_indexes_to_load: Union[List[TessTwoMinuteCadenceMastFitsIndex], None] = None
                  ) -> TessTwoMinuteCadenceFileBasedLightCurve:
        """
        Creates a TESS two minute light curve from a path to the MAST FITS file..

        :param path: The path to the FITS file to load.
        :param flux_column_name: The column name to use for the default flux attribute.
        :param fits_indexes_to_load: The indexes to load from the FITS file. By default, all will be loaded. Selecting
                                     specific ones may speed the process when loading many ligh tcurves.
        :return: The light curve.
        """
        light_curve = cls()
        light_curve.flux_column_name = flux_column_name
        light_curve.time_column_name = TessTwoMinuteCadenceColumnName.TIME.value
        if fits_indexes_to_load is None:
            fits_indexes_to_load = list(TessTwoMinuteCadenceColumnName)
        with fits.open(path) as hdu_list:
            light_curve_table = hdu_list[1].data  # Lightcurve information is in first extension table.
            for fits_index in fits_indexes_to_load:
                column_name = TessTwoMinuteCadenceColumnName[fits_index.name]
                light_curve.data_frame[column_name.value] = light_curve_table[fits_index.value]
        return light_curve
