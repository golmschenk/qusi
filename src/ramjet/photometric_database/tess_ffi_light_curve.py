"""
Code to for a class to represent a TESS FFI light curve.
"""
from __future__ import annotations

import os
import pickle
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astroquery.vizier import Vizier
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from ramjet.photometric_database.tess_light_curve import TessLightCurve

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class TessFfiColumnName(Enum):
    """
    An enum to represent the column names of the TESS FFI data.
    """

    TIME__BTJD = "time__btjd"
    CORRECTED_FLUX = "corrected_flux"
    RAW_FLUX = "raw_flux"
    FLUX_ERROR = "flux_error"
    QUALITY_FLAG = "quality_flag"


class TessFfiPickleIndex(Enum):
    """
    An enum for accessing Brian Powell's FFI pickle data with understandable index names.
    """

    TIC_ID = 0
    RA = 1
    DEC = 2
    TESS_MAGNITUDE = 3
    CAMERA = 4
    CHIP = 5
    TIME__BTJD = 6
    RAW_FLUX = 7
    CORRECTED_FLUX = 8
    PCA_FLUX = 9
    FLUX_ERROR = 10
    QUALITY_FLAG = 11


class AdaptIntermittentError(Exception):
    pass


def adapt_intermittent_error(exception: Exception) -> bool:
    return isinstance(exception, (OSError, pickle.UnpicklingError))


class TessFfiLightCurve(TessLightCurve):
    """
    A class to for a class to represent a TESS FFI light curve.
    """

    def __init__(self):
        super().__init__()
        self.flux_column_names = [TessFfiColumnName.CORRECTED_FLUX.value, TessFfiColumnName.RAW_FLUX.value]

    @classmethod
    @retry(
        retry=retry_if_exception_type(AdaptIntermittentError),
        wait=wait_random_exponential(multiplier=0.1, max=20),
        stop=stop_after_attempt(20),
        reraise=True,
    )
    def from_path(
        cls,
        path: Path,
        column_names_to_load: list[TessFfiColumnName] | None = None,
        *,
        remove_bad_quality_data: bool = True,
    ) -> TessFfiLightCurve:
        """
        Creates an FFI TESS light curve from a path to one of Brian Powell's pickle files.

        :param path: The path to the pickle file to load.
        :param column_names_to_load: The FFI light curve columns to load from the pickle file. By default, all will be
                                     loaded. Selecting specific ones may speed the process when loading many light
                                     curves.
        :param remove_bad_quality_data: Removes data with quality problem flags (e.g., non-zero quality flags).
        :return: The light curve.
        """
        light_curve = cls()
        light_curve.time_column_name = TessFfiColumnName.TIME__BTJD.value
        if column_names_to_load is None:
            column_names_to_load = list(TessFfiColumnName)
        try:
            with path.open("rb") as pickle_file:
                light_curve_data_dictionary = pickle.load(pickle_file)
        except (pickle.UnpicklingError, OSError, IsADirectoryError) as error:
            error_message = f"Errored on path {path}."
            raise AdaptIntermittentError(error_message) from error
        if remove_bad_quality_data:
            quality_flag_values = light_curve_data_dictionary[TessFfiPickleIndex.QUALITY_FLAG.value]
            for column_name in column_names_to_load:
                pickle_index = TessFfiPickleIndex[column_name.name]
                column_values = light_curve_data_dictionary[pickle_index.value]
                if remove_bad_quality_data:
                    column_values = column_values[quality_flag_values == 0]
                light_curve.data_frame[column_name.value] = column_values
        light_curve.tic_id, light_curve.sector = light_curve.get_tic_id_and_sector_from_file_path(path)
        return light_curve

    @staticmethod
    def get_tic_id_and_sector_from_file_path(path: Path | str) -> (int, int | None):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        path = str(Path(path))
        sep_str = os.sep
        if sep_str == "\\":
            sep_str = "\\\\"
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(
            rf"tesslcs_sector_(\d+)(?:_104)?{sep_str}(?:2_min_cadence_targets|tesslcs_tmag_\d+_\d+){sep_str}tesslc_(\d+)",
            path,
        )
        if match:
            return int(match.group(2)), int(match.group(1))
        # Search for Brian Powell's FFI path convention with only the file name containing the target.
        # E.g., "tesslc_290374453"
        match = re.search(r"tesslc_(\d+)", path)
        if match:
            return int(match.group(1)), None
        # Search for project specific rename of Brian Powell's FFI path convention for flat directory.
        match = re.search(r"tic_id_(\d+)_sector_(\d+)_ffi_light_curve.pkl", path)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Raise an error if none of the patterns matched.
        error_message = f"{path} does not match a known pattern to extract TIC ID and sector from."
        raise ValueError(error_message)

    @staticmethod
    def get_floor_magnitude_from_file_path(file_path: Path | str) -> int:
        """
        Gets the floor magnitude from the FFI file path.

        :param file_path: The path of the file to extract the magnitude.
        :return: The magnitude floored.
        """
        file_path_string = str(Path(file_path))
        sep_str = os.sep
        if sep_str == "\\":
            sep_str = "\\\\"
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(
            rf"tesslcs_sector_\d+(?:_104)?{sep_str}tesslcs_tmag_(\d+)_\d+{sep_str}tesslc_\d+", file_path_string
        )
        if match:
            return int(match.group(1))
        error_message = f"{file_path_string} does not match a known pattern to extract magnitude from."
        raise ValueError(error_message)

    @staticmethod
    def get_magnitude_from_file(file_path: Path | str) -> float:
        """
        Loads the magnitude from the file.

        :param file_path: The path to the file.
        :return: The magnitude of the target.
        """
        with file_path.open("rb") as pickle_file:
            light_curve = pickle.load(pickle_file)
        magnitude = light_curve[TessFfiPickleIndex.TESS_MAGNITUDE.value]
        return magnitude

    @classmethod
    def load_fluxes_and_times_from_pickle_file(
        cls,
        file_path: Path | str,
        flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX,
        *,
        remove_bad_quality_data: bool = True,
    ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_column_name: The flux type to load.
        :param remove_bad_quality_data: Removes data with quality problem flags (e.g., non-zero quality flags).
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        columns_to_load = [TessFfiColumnName.TIME__BTJD, flux_column_name]
        light_curve = cls.from_path(
            file_path, column_names_to_load=columns_to_load, remove_bad_quality_data=remove_bad_quality_data
        )
        fluxes = light_curve.data_frame[flux_column_name.value]
        times = light_curve.data_frame[TessFfiColumnName.TIME__BTJD.value]
        if times.shape != fluxes.shape:
            error_message = (
                f"Times and fluxes arrays must have the same shape, but have shapes "
                f"{times.shape} and {fluxes.shape}."
            )
            raise ValueError(error_message)
        return fluxes, times

    @classmethod
    def load_fluxes_flux_errors_and_times_from_pickle_file(
        cls, file_path: Path | str, flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Loads the fluxes, flux errors, and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_column_name: The flux type to load.
        :return: The fluxes, flux errors, and times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        light_curve = cls.from_path(
            file_path,
            column_names_to_load=[TessFfiColumnName.TIME__BTJD, TessFfiColumnName.FLUX_ERROR, flux_column_name],
        )
        fluxes = light_curve.data_frame[flux_column_name.value]
        flux_errors = light_curve.data_frame[TessFfiColumnName.FLUX_ERROR.value]
        times = light_curve.data_frame[TessFfiColumnName.TIME__BTJD.value]
        if times.shape != fluxes.shape:
            error_message = (
                f"Times and fluxes arrays must have the same shape, but have shapes "
                f"{times.shape} and {fluxes.shape}."
            )
            raise ValueError(error_message)
        return fluxes, flux_errors, times


class GcvsColumnName(StrEnum):
    VARIABLE_TYPE_STRING = "VarType"
    RA = "RAJ2000"
    DEC = "DEJ2000"


def has_gcvs_type(var_type_string: str, labels: list[str]) -> bool:
    var_type_string_without_uncertainty_flags = var_type_string.replace(":", "")
    variable_type_flags = var_type_string_without_uncertainty_flags.split("+")
    return any(variable_type_flag in labels for variable_type_flag in variable_type_flags)


def get_gcvs_catalog_entries_for_labels(labels: list[str]) -> pd.DataFrame:
    # TODO: Not keeping this function, just copying stuff from it.
    gcvs_catalog_astropy_table = Vizier(columns=["**"], catalog="B/gcvs/gcvs_cat", row_limit=-1).query_constraints()[0]
    gcvs_catalog_data_frame = gcvs_catalog_astropy_table.to_pandas()

    def filter_function(var_type_string):
        return has_gcvs_type(var_type_string, labels)

    label_mask = gcvs_catalog_data_frame[GcvsColumnName.VARIABLE_TYPE_STRING].apply(filter_function)
    data_frame_of_classes = gcvs_catalog_data_frame[label_mask]
    return data_frame_of_classes


def separation_to_nearest_gcvs_rr_lyrae_within_separation(
    sky_coord: SkyCoord, maximum_separation: Angle | None = None
) -> Angle | None:
    if maximum_separation is None:
        maximum_separation = Angle(21, unit=units.arcsecond)
    gcvs_region_table_list = Vizier(columns=["**"], catalog="B/gcvs/gcvs_cat", row_limit=-1).query_region(
        sky_coord, radius=maximum_separation
    )
    if len(gcvs_region_table_list) == 0:
        return None
    gcvs_region_data_frame = gcvs_region_table_list[0].to_pandas()
    rr_lyrae_labels = ["RR", "RR(B)", "RRAB", "RRC"]

    def filter_function(var_type_string):
        return has_gcvs_type(var_type_string, rr_lyrae_labels)

    label_mask = gcvs_region_data_frame[GcvsColumnName.VARIABLE_TYPE_STRING].apply(filter_function)
    rr_lyrae_region_data_frame = gcvs_region_data_frame[label_mask]
    try:
        closet_rr_lyrae_row = rr_lyrae_region_data_frame.iloc[0]
        closet_rr_lyrae_coordinates = SkyCoord(
            ra=closet_rr_lyrae_row["RAJ2000"],
            dec=closet_rr_lyrae_row["DEJ2000"],
            unit=(units.hourangle, units.deg),
            equinox="J2000",
        )
        return sky_coord.separation(closet_rr_lyrae_coordinates)
    except IndexError:
        return None


tess_pixel_angular_size = Angle(21, unit=units.arcsecond)
