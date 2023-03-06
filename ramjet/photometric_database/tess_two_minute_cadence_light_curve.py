"""
Code to for a class to represent a TESS two minute cadence light curve.
"""
from __future__ import annotations

import re
import numpy as np
from enum import Enum
from pathlib import Path
from typing import List, Union, Any, Optional
from astropy.io import fits

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.photometric_database.tess_light_curve import TessLightCurve


class TessTwoMinuteCadenceColumnName(Enum):
    """
    An enum to represent the column names of the TESS two minute cadence data.
    """
    TIME__BTJD = 'time__btjd'
    SAP_FLUX = 'sap_flux'
    PDCSAP_FLUX = 'pdcsap_flux'
    SAP_FLUX_ERROR = 'sap_flux_error'
    PDCSAP_FLUX_ERROR = 'pdcsap_flux_error'


class TessTwoMinuteCadenceMastFitsIndex(Enum):
    """
    An enum to represent the indexes of the TESS two minute cadence data in MAST FITS files.
    """
    TIME__BTJD = 'TIME'
    SAP_FLUX = 'SAP_FLUX'
    PDCSAP_FLUX = 'PDCSAP_FLUX'
    SAP_FLUX_ERROR = 'SAP_FLUX_ERR'
    PDCSAP_FLUX_ERROR = 'PDCSAP_FLUX_ERR'


class TessTwoMinuteCadenceLightCurve(TessLightCurve):
    """
    A class to represent a TESS two minute cadence light curve.
    """
    mast_tess_data_interface = TessDataInterface()

    def __init__(self):
        super().__init__()
        self.flux_column_names = [TessTwoMinuteCadenceColumnName.PDCSAP_FLUX.value,
                                  TessTwoMinuteCadenceColumnName.SAP_FLUX.value]

    @classmethod
    def from_path(cls, path: Path, fits_indexes_to_load: Union[List[TessTwoMinuteCadenceMastFitsIndex], None] = None
                  ) -> TessTwoMinuteCadenceLightCurve:
        """
        Creates a TESS two minute light curve from a path to the MAST FITS file.

        :param path: The path to the FITS file to load.
        :param fits_indexes_to_load: The indexes to load from the FITS file. By default, all will be loaded. Selecting
                                     specific ones may speed the process when loading many light curves.
        :return: The light curve.
        """
        light_curve = cls()
        light_curve.time_column_name = TessTwoMinuteCadenceColumnName.TIME__BTJD.value
        if fits_indexes_to_load is None:
            fits_indexes_to_load = list(TessTwoMinuteCadenceMastFitsIndex)
        with fits.open(path) as hdu_list:
            light_curve_table = hdu_list[1].data  # Light curve information is in first extension table.
            for fits_index in fits_indexes_to_load:
                column_name = TessTwoMinuteCadenceColumnName[fits_index.name]
                light_curve.data_frame[column_name.value] = light_curve_table[fits_index.value]
        light_curve.tic_id, light_curve.sector = cls.get_tic_id_and_sector_from_file_path(path)
        return light_curve

    @classmethod
    def from_mast(cls, tic_id: int, sector: Optional[int] = None,
                  fits_indexes_to_load: Union[List[TessTwoMinuteCadenceMastFitsIndex], None] = None
                  ) -> TessTwoMinuteCadenceLightCurve:
        """
        Downloads a FITS file from MAST and creates a TESS two minute light curve from it.

        :param tic_id: The TIC ID of the target.
        :param sector: The sector of the observation.
        :param fits_indexes_to_load: The indexes to load from the FITS file. By default, all will be loaded. Selecting
                                     specific ones may speed the process when loading many light curves.
        :return: The light curve.
        """
        light_curve_path = cls.mast_tess_data_interface.download_two_minute_cadence_light_curve(tic_id=tic_id,
                                                                                                sector=sector)
        light_curve = cls.from_path(path=light_curve_path, fits_indexes_to_load=fits_indexes_to_load)
        return light_curve

    @classmethod
    def from_identifier(cls, identifier: Any) -> TessTwoMinuteCadenceLightCurve:
        """
        Loads the light curve in a generalized way, attempting to infer the light curve based on the passed identifier.

        :param identifier: The identifier of the light curve. Could come in various forms.
        :return: The light curve.
        """
        integer_types = (int, np.integer)
        if isinstance(identifier, Path):
            return cls.from_path(path=identifier)
        elif isinstance(identifier, tuple) and (isinstance(identifier[0], integer_types) and
                                                isinstance(identifier[1], integer_types)):
            tic_id = identifier[0]
            sector = identifier[1]
            return cls.from_mast(tic_id=tic_id, sector=sector)
        elif isinstance(identifier, str):
            tic_id, sector = cls.get_tic_id_and_sector_from_identifier_string(identifier)
            return cls.from_mast(tic_id=tic_id, sector=sector)
        else:
            raise ValueError(f'{identifier} does not match a known type to infer the light curve identifier from.')

    @staticmethod
    def get_tic_id_and_sector_from_file_path(file_path: Path) -> (int, Union[int, None]):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param file_path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        file_name = file_path.stem
        tic_id, sector = TessTwoMinuteCadenceLightCurve.get_tic_id_and_sector_from_identifier_string(file_name)
        return tic_id, sector

    @staticmethod
    def get_tic_id_and_sector_from_identifier_string(identifier_string: str) -> (int, Union[int, None]):
        """
        Gets the TIC ID and sector from commonly encountered identifier string patterns.

        :param identifier_string: The string to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        # Search for the human readable version. E.g., "TIC 169480782 sector 5"
        match = re.search(r'TIC (\d+) sector (\d+)', identifier_string)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Search for the human readable TIC only version. E.g., "TIC 169480782"
        match = re.search(r'TIC (\d+)', identifier_string)
        if match:
            return int(match.group(1)), None
        # Search for the TESS obs_id version. E.g., "tess2018319095959-s0005-0000000278956474-0125-s"
        match = re.search(r'tess\d+-s(\d+)-(\d+)-\d+-s', identifier_string)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Raise an error if none of the patterns matched.
        raise ValueError(f'{identifier_string} does not match a known pattern to extract TIC ID and sector from.')
