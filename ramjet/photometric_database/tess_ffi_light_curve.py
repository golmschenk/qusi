"""
Code to for a class to represent a TESS FFI light curve.
"""
from __future__ import annotations

import copy
import pickle
import re

import lightkurve
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Union, List, Optional

import pandas as pd
from astropy import units
from astropy.coordinates import SkyCoord, Angle
from astroquery.vizier import Vizier
from retrying import retry

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum
from lightkurve.targetpixelfile import TargetPixelFile

from ramjet.data_interface.tess_data_interface import TessDataInterface, is_common_mast_connection_error
from ramjet.photometric_database.tess_light_curve import TessLightCurve


class TessFfiColumnName(Enum):
    """
    An enum to represent the column names of the TESS FFI data.
    """
    TIME__BTJD = 'time__btjd'
    CORRECTED_FLUX = 'corrected_flux'
    RAW_FLUX = 'raw_flux'
    FLUX_ERROR = 'flux_error'
    QUALITY_FLAG = 'quality_flag'


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


class CentroidAlgorithmFailedError(Exception):
    pass


class TessFfiLightCurve(TessLightCurve):
    """
    A class to for a class to represent a TESS FFI light curve.
    """
    mast_tess_data_interface = TessDataInterface()

    def __init__(self):
        super().__init__()
        self.flux_column_names = [TessFfiColumnName.CORRECTED_FLUX.value,
                                  TessFfiColumnName.RAW_FLUX.value]

    @classmethod
    def from_path(cls, path: Path, column_names_to_load: Union[List[TessFfiColumnName], None] = None,
                  remove_bad_quality_data: bool = True) -> TessFfiLightCurve:
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
        with path.open('rb') as pickle_file:
            light_curve_data_dictionary = pickle.load(pickle_file)
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
    def get_tic_id_and_sector_from_file_path(path: Union[Path, str]) -> (int, Union[int, None]):
        """
        Gets the TIC ID and sector from commonly encountered file name patterns.

        :param path: The path of the file to extract the TIC ID and sector.
        :return: The TIC ID and sector. The sector might be omitted (as None).
        """
        if isinstance(path, Path):
            path = str(path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_(\d+)(?:_104)?/(?:2_min_cadence_targets|tesslcs_tmag_\d+_\d+)/tesslc_(\d+)', path)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Search for Brian Powell's FFI path convention with only the file name containing the target.
        # E.g., "tesslc_290374453"
        match = re.search(r'tesslc_(\d+)', path)
        if match:
            return int(match.group(1)), None
        # Search for project specific rename of Brian Powell's FFI path convention for flat directory.
        match = re.search(r'tic_id_(\d+)_sector_(\d+)_ffi_light_curve.pkl', path)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Raise an error if none of the patterns matched.
        raise ValueError(f'{path} does not match a known pattern to extract TIC ID and sector from.')

    @staticmethod
    def get_floor_magnitude_from_file_path(file_path: Union[Path, str]) -> int:
        """
        Gets the floor magnitude from the FFI file path.

        :param file_path: The path of the file to extract the magnitude.
        :return: The magnitude floored.
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)
        # Search for Brian Powell's FFI path convention with directory structure sector, magnitude, target.
        # E.g., "tesslcs_sector_12/tesslcs_tmag_1_2/tesslc_290374453"
        match = re.search(r'tesslcs_sector_\d+(?:_104)?/tesslcs_tmag_(\d+)_\d+/tesslc_\d+', file_path)
        if match:
            return int(match.group(1))
        raise ValueError(f'{file_path} does not match a known pattern to extract magnitude from.')

    @staticmethod
    def get_magnitude_from_file(file_path: Union[Path, str]) -> float:
        """
        Loads the magnitude from the file.

        :param file_path: The path to the file.
        :return: The magnitude of the target.
        """
        with file_path.open('rb') as pickle_file:
            light_curve = pickle.load(pickle_file)
        magnitude = light_curve[TessFfiPickleIndex.TESS_MAGNITUDE.value]
        return magnitude

    @classmethod
    def load_fluxes_and_times_from_pickle_file(
            cls, file_path: Union[Path, str], flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX,
            remove_bad_quality_data: bool = True
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
        columns_to_load = [TessFfiColumnName.TIME__BTJD,
                           flux_column_name]
        light_curve = cls.from_path(file_path, column_names_to_load=columns_to_load,
                                    remove_bad_quality_data=remove_bad_quality_data)
        fluxes = light_curve.data_frame[flux_column_name.value]
        times = light_curve.data_frame[TessFfiColumnName.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        return fluxes, times

    @classmethod
    def load_fluxes_flux_errors_and_times_from_pickle_file(
            cls, file_path: Union[Path, str], flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Loads the fluxes, flux errors, and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_column_name: The flux type to load.
        :return: The fluxes, flux errors, and times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        light_curve = cls.from_path(file_path, column_names_to_load=[TessFfiColumnName.TIME__BTJD,
                                                                     TessFfiColumnName.FLUX_ERROR,
                                                                     flux_column_name])
        fluxes = light_curve.data_frame[flux_column_name.value]
        flux_errors = light_curve.data_frame[TessFfiColumnName.FLUX_ERROR.value]
        times = light_curve.data_frame[TessFfiColumnName.TIME__BTJD.value]
        assert times.shape == fluxes.shape
        return fluxes, flux_errors, times

    @retry(retry_on_exception=is_common_mast_connection_error)
    def get_ffi_time_series_from_tess_cut(self) -> TargetPixelFile:
        search_result = lightkurve.search_tesscut(f'TIC{self.tic_id}', sector=self.sector)
        target_pixel_file = search_result.download(cutout_size=10)
        return target_pixel_file

    def get_photometric_centroid_of_variability(self, minimum_period: Optional[float] = None,
                                                maximum_period: Optional[float] = None) -> SkyCoord:
        fold_period, fold_epoch, time_bin_size, minimum_bin_phase, maximum_bin_phase = \
            self.get_variability_phase_folding_parameters(minimum_period=minimum_period, maximum_period=maximum_period)
        variability_centroid_and_frames = \
            self.get_photometric_variability_centroid_and_frames_from_folding_parameters(fold_epoch,
                                                                                         fold_period,
                                                                                         maximum_bin_phase,
                                                                                         minimum_bin_phase,
                                                                                         time_bin_size)
        centroid_sky_coord = variability_centroid_and_frames[0]
        return centroid_sky_coord

    def get_photometric_variability_centroid_and_frames_from_folding_parameters(
            self, fold_epoch: float, fold_period: float, maximum_bin_phase: float, minimum_bin_phase: float,
            time_bin_size: float):
        target_pixel_file = self.get_ffi_time_series_from_tess_cut()
        if target_pixel_file is None:
            raise CentroidAlgorithmFailedError
        phases = ((target_pixel_file.time.value - fold_epoch) % fold_period)
        minimum_bin_indexes = np.where((phases > (minimum_bin_phase - time_bin_size)) &
                                       (phases < (minimum_bin_phase + time_bin_size)))
        maximum_bin_indexes = np.where((phases > (maximum_bin_phase - time_bin_size)) &
                                       (phases < (maximum_bin_phase + time_bin_size)))
        # Hack to get a single frame target pixel file with the right coordinates, etc. by copying the first frame.
        median_minimum_target_pixel_frame = copy.deepcopy(target_pixel_file[minimum_bin_indexes][0])
        median_minimum_target_pixel_frame.hdu[1].data["FLUX"] = np.nanmedian(
            target_pixel_file[minimum_bin_indexes].flux.value, axis=0, keepdims=True)
        # Hack to get a single frame target pixel file with the right coordinates, etc. by copying the first frame.
        median_maximum_target_pixel_frame = copy.deepcopy(target_pixel_file[maximum_bin_indexes][0])
        median_maximum_target_pixel_frame.hdu[1].data["FLUX"] = np.nanmedian(
            target_pixel_file[maximum_bin_indexes].flux.value, axis=0, keepdims=True)
        # Hack to get a single frame target pixel file with the right coordinates, etc. by copying the first frame.
        difference_target_pixel_frame = copy.deepcopy(target_pixel_file[0])
        difference_flux_frame = median_maximum_target_pixel_frame.flux.value - median_minimum_target_pixel_frame.flux.value
        difference_target_pixel_frame.hdu[1].data["FLUX"] = difference_flux_frame
        image_side_size = 10
        pixel_side_indexes = np.arange(image_side_size, dtype=np.float32)
        flux_difference = difference_flux_frame[0]
        positive_flux_difference = np.maximum(flux_difference, 0)
        try:
            x_flux_difference_centroid = np.average(pixel_side_indexes,
                                                    weights=np.mean(positive_flux_difference, axis=0))
            y_flux_difference_centroid = np.average(pixel_side_indexes,
                                                    weights=np.mean(positive_flux_difference, axis=1))
            centroid_sky_coord = target_pixel_file.wcs.pixel_to_world(x_flux_difference_centroid,
                                                                      y_flux_difference_centroid)
        except ZeroDivisionError as error:
            raise CentroidAlgorithmFailedError from error
        return centroid_sky_coord, target_pixel_file, difference_target_pixel_frame, median_maximum_target_pixel_frame, median_minimum_target_pixel_frame

    def get_angular_distance_to_variability_photometric_centroid(self, minimum_period: Optional[float] = None,
                                                                 maximum_period: Optional[float] = None) -> Angle:
        centroid_sky_coord = self.get_photometric_centroid_of_variability(minimum_period=minimum_period,
                                                                          maximum_period=maximum_period)
        return self.sky_coord.separation(centroid_sky_coord)


class GcvsColumnName(StrEnum):
    VARIABLE_TYPE_STRING = 'VarType'
    RA = 'RAJ2000'
    DEC = 'DEJ2000'


def has_gcvs_type(var_type_string: str, labels: List[str]) -> bool:
    var_type_string_without_uncertainty_flags = var_type_string.replace(':', '')
    variable_type_flags = var_type_string_without_uncertainty_flags.split('+')
    for variable_type_flag in variable_type_flags:
        if variable_type_flag in labels:
            return True
    return False


def get_gcvs_catalog_entries_for_labels(labels: List[str]) -> pd.DataFrame:
    # TODO: Not keeping this function, just copying stuff from it.
    gcvs_catalog_astropy_table = Vizier(columns=['**'], catalog='B/gcvs/gcvs_cat', row_limit=-1).query_constraints()[0]
    gcvs_catalog_data_frame = gcvs_catalog_astropy_table.to_pandas()

    def filter_function(var_type_string):
        return has_gcvs_type(var_type_string, labels)

    label_mask = gcvs_catalog_data_frame[GcvsColumnName.VARIABLE_TYPE_STRING].apply(filter_function)
    data_frame_of_classes = gcvs_catalog_data_frame[label_mask]
    return data_frame_of_classes


def separation_to_nearest_gcvs_rr_lyrae_within_separation(sky_coord: SkyCoord,
                                                          maximum_separation: Angle(21, unit=units.arcsecond)
                                                          ) -> Optional[Angle]:
    gcvs_region_table_list = Vizier(columns=['**'], catalog='B/gcvs/gcvs_cat', row_limit=-1
                                    ).query_region(sky_coord, radius=maximum_separation)
    if len(gcvs_region_table_list) == 0:
        return None
    gcvs_region_data_frame = gcvs_region_table_list[0].to_pandas()
    rr_lyrae_labels = ['RR', 'RR(B)', 'RRAB', 'RRC']

    def filter_function(var_type_string):
        return has_gcvs_type(var_type_string, rr_lyrae_labels)

    label_mask = gcvs_region_data_frame[GcvsColumnName.VARIABLE_TYPE_STRING].apply(filter_function)
    rr_lyrae_region_data_frame = gcvs_region_data_frame[label_mask]
    try:
        closet_rr_lyrae_row = rr_lyrae_region_data_frame.iloc[0]
        closet_rr_lyrae_coordinates = SkyCoord(ra=closet_rr_lyrae_row['RAJ2000'], dec=closet_rr_lyrae_row['DEJ2000'],
                                               unit=(units.hourangle, units.deg), equinox='J2000')
        return sky_coord.separation(closet_rr_lyrae_coordinates)
    except IndexError:
        return None


tess_pixel_angular_size = Angle(21, unit=units.arcsecond)
