"""
Code to for a class to represent a TESS two minute cadence light curve.
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import List, Union
from astropy.io import fits

from ramjet.data_interface.tess_data_interface import TessDataInterface
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


class TessTwoMinuteCadenceLightCurve(LightCurve):
    """
    A class to represent a TESS two minute cadence light curve.
    """
    mast_tess_data_interface = TessDataInterface()
    flux_column_names = [TessTwoMinuteCadenceColumnName.PDCSAP_FLUX.value,
                         TessTwoMinuteCadenceColumnName.SAP_FLUX.value]

    def __init__(self):
        super().__init__()
        self.tic_id: Union[int, None] = None
        self.sector: Union[int, None] = None

    @classmethod
    def from_path(cls, path: Path,
                  flux_column_name: TessTwoMinuteCadenceColumnName = TessTwoMinuteCadenceColumnName.PDCSAP_FLUX.value,
                  fits_indexes_to_load: Union[List[TessTwoMinuteCadenceMastFitsIndex], None] = None
                  ) -> TessTwoMinuteCadenceLightCurve:
        """
        Creates a TESS two minute light curve from a path to the MAST FITS file.

        :param path: The path to the FITS file to load.
        :param flux_column_name: The column name to use for the default flux attribute.
        :param fits_indexes_to_load: The indexes to load from the FITS file. By default, all will be loaded. Selecting
                                     specific ones may speed the process when loading many light curves.
        :return: The light curve.
        """
        light_curve = cls()
        light_curve.flux_column_name = flux_column_name
        light_curve.time_column_name = TessTwoMinuteCadenceColumnName.TIME.value
        if fits_indexes_to_load is None:
            fits_indexes_to_load = list(TessTwoMinuteCadenceMastFitsIndex)
        with fits.open(path) as hdu_list:
            light_curve_table = hdu_list[1].data  # Lightcurve information is in first extension table.
            for fits_index in fits_indexes_to_load:
                column_name = TessTwoMinuteCadenceColumnName[fits_index.name]
                light_curve.data_frame[column_name.value] = light_curve_table[fits_index.value]
        light_curve.tic_id, light_curve.sector = cls.get_tic_id_and_sector_from_file_path(path)
        return light_curve

    @classmethod
    def from_mast(cls, tic_id:int, sector: int,
                  flux_column_name: TessTwoMinuteCadenceColumnName = TessTwoMinuteCadenceColumnName.PDCSAP_FLUX.value,
                  fits_indexes_to_load: Union[List[TessTwoMinuteCadenceMastFitsIndex], None] = None
                  ) -> TessTwoMinuteCadenceLightCurve:
        """
        Downloads a FITS file from MAST and creates a TESS two minute light curve from it.

        :param tic_id: The TIC ID of the target.
        :param sector: The sector of the observation.
        :param flux_column_name: The column name to use for the default flux attribute.
        :param fits_indexes_to_load: The indexes to load from the FITS file. By default, all will be loaded. Selecting
                                     specific ones may speed the process when loading many light curves.
        :return: The light curve.
        """
        light_curve_path = cls.mast_tess_data_interface.download_lightcurve(tic_id=tic_id, sector=sector)
        light_curve = cls.from_path(path=light_curve_path, flux_column_name=flux_column_name,
                                    fits_indexes_to_load=fits_indexes_to_load)
        return light_curve

    @staticmethod
    def get_tic_id_and_sector_from_file_path(file_path: Union[Path, str]):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param file_path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        file_name = file_path.stem
        # Search for the human readable version. E.g., "TIC 169480782 sector 5"
        match = re.search(r'TIC (\d+) sector (\d+)', file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Search for the TESS obs_id version. E.g., "tess2018319095959-s0005-0000000278956474-0125-s"
        match = re.search(r'tess\d+-s(\d+)-(\d+)-\d+-s', file_name)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Raise an error if none of the patterns matched.
        raise ValueError(f'{file_name} does not match a known pattern to extract TIC ID and sector from.')
