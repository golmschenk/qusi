"""
Code to for a class to represent a TESS FFI light curve.
"""
from __future__ import annotations

import copy
import pickle
import re

import astropy
import lightkurve
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Union, List

from astropy import units
from astropy.coordinates import SkyCoord, Angle
from astroquery.mast import Catalogs
from lightkurve.targetpixelfile import TargetPixelFile

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.photometric_database.tess_light_curve import TessLightCurve


class TessFfiColumnName(Enum):
    """
    An enum to represent the column names of the TESS FFI data.
    """
    TIME__BTJD = 'time__btjd'
    CORRECTED_FLUX = 'corrected_flux'
    RAW_FLUX = 'raw_flux'
    FLUX_ERROR = 'flux_error'


class TessFfiPickleIndex(Enum):
    """
    An enum for accessing Brian Powell's FFI pickle data with understandable indexes.
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
    QUALITY = 11


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
    def from_path(cls, path: Path, column_names_to_load: Union[List[TessFfiColumnName], None] = None
                  ) -> TessFfiLightCurve:
        """
        Creates an FFI TESS light curve from a path to one of Brian Powell's pickle files.

        :param path: The path to the pickle file to load.
        :param column_names_to_load: The FFI light curve columns to load from the pickle file. By default, all will be
                                     loaded. Selecting specific ones may speed the process when loading many light
                                     curves.
        :return: The light curve.
        """
        light_curve = cls()
        light_curve.time_column_name = TessFfiColumnName.TIME__BTJD.value
        if column_names_to_load is None:
            column_names_to_load = list(TessFfiColumnName)
        with path.open('rb') as pickle_file:
            light_curve_data_dictionary = pickle.load(pickle_file)
            for column_name in column_names_to_load:
                pickle_index = TessFfiPickleIndex[column_name.name]
                light_curve.data_frame[column_name.value] = light_curve_data_dictionary[pickle_index.value]
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
        match = re.search(r'tesslcs_sector_(\d+)(?:_104)?/tesslcs_tmag_\d+_\d+/tesslc_(\d+)', path)
        if match:
            return int(match.group(2)), int(match.group(1))
        # Search for Brian Powell's FFI path convention with only the file name containing the target.
        # E.g., "tesslc_290374453"
        match = re.search(r'tesslc_(\d+)', path)
        if match:
            return int(match.group(1)), None
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
            cls, file_path: Union[Path, str], flux_column_name: TessFfiColumnName = TessFfiColumnName.CORRECTED_FLUX
        ) -> (np.ndarray, np.ndarray):
        """
        Loads the fluxes and times from one of Brian Powell's FFI pickle files.

        :param file_path: The path to the pickle file to load.
        :param flux_column_name: The flux type to load.
        :return: The fluxes and the times.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        light_curve = cls.from_path(file_path, column_names_to_load=[TessFfiColumnName.TIME__BTJD,
                                                                     flux_column_name])
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

    def get_ffi_time_series_from_tess_cut(self) -> TargetPixelFile:
        search_result = lightkurve.search_tesscut(f'TIC{self.tic_id}', sector=self.sector)
        target_pixel_file = search_result.download(cutout_size=10)
        return target_pixel_file
        # target_pixel_file.plot()
        # from matplotlib import pyplot as plt
        # sky_coord = SkyCoord(ra=target_pixel_file.ra, dec=target_pixel_file.dec, unit=units.deg)
        # pixel_position = target_pixel_file.wcs.world_to_pixel(sky_coord)
        # plt.plot(pixel_position[0] + target_pixel_file.column, pixel_position[1] + target_pixel_file.row, 'ro')
        # tess_pixel_fov = Angle(21, unit=units.arcsecond)
        # region_results = Catalogs.query_region(sky_coord, radius=tess_pixel_fov * 5, catalog='TIC',
        #                                        columns=['**']).to_pandas().sort_values('Tmag')
        # brightest_neighbor_row = region_results.iloc[0]
        # brightest_neighbor_sky_coord = SkyCoord(ra=brightest_neighbor_row['ra'], dec=brightest_neighbor_row['dec'], unit=units.deg)
        # brightest_neighbor_pixel_position = target_pixel_file.wcs.world_to_pixel(brightest_neighbor_sky_coord)
        # plt.plot(brightest_neighbor_pixel_position[0] + target_pixel_file.column, brightest_neighbor_pixel_position[1] + target_pixel_file.row, 'mo')
        # plt.plot(target_pixel_file.column, target_pixel_file.row, 'mo')
        # plt.plot(target_pixel_file.column + 4.5, target_pixel_file.row + 4.5, 'mo')
        # five_five_coord = target_pixel_file.wcs.pixel_to_world(5, 5)
        # plt.show()
        # pass

    def get_photometric_centroid_of_variability(self) -> SkyCoord:
        fold_period, fold_epoch, time_bin_size, minimum_bin_phase, maximum_bin_phase = \
            self.get_variability_phase_folding_parameters()
        target_pixel_file = self.get_ffi_time_series_from_tess_cut()
        phases = ((target_pixel_file.time.value - fold_epoch) % fold_period)
        minimum_bin_indexes = np.where((phases > (minimum_bin_phase - time_bin_size)) &
                                       (phases < (minimum_bin_phase + time_bin_size)))
        maximum_bin_indexes = np.where((phases > (maximum_bin_phase - time_bin_size)) &
                                       (phases < (maximum_bin_phase + time_bin_size)))
        # Hack to get a single frame target pixel file with the right coordinates, etc.
        from matplotlib import pyplot as plt
        median_minimum_target_pixel_frame = copy.deepcopy(target_pixel_file[minimum_bin_indexes][0])
        median_minimum_target_pixel_frame.hdu[1].data["FLUX"] = np.nanmedian(target_pixel_file[minimum_bin_indexes].flux.value, axis=0, keepdims=True)
        # median_minimum_target_pixel_frame.plot()
        # plt.show()
        median_maximum_target_pixel_frame = copy.deepcopy(target_pixel_file[maximum_bin_indexes][0])
        median_maximum_target_pixel_frame.hdu[1].data["FLUX"] = np.nanmedian(target_pixel_file[maximum_bin_indexes].flux.value, axis=0, keepdims=True)
        # median_maximum_target_pixel_frame.plot()
        # plt.show()
        difference_target_pixel_frame = copy.deepcopy(target_pixel_file[0])
        difference_flux_frame = median_maximum_target_pixel_frame.flux.value - median_minimum_target_pixel_frame.flux.value
        # difference_flux_frame = difference_flux_frame - difference_flux_frame.min()
        difference_target_pixel_frame.hdu[1].data["FLUX"] = difference_flux_frame
        # difference_target_pixel_frame.plot()
        image_side_size = 10
        pixel_side_indexes = np.arange(image_side_size, dtype=np.float32)
        flux_difference = difference_flux_frame[0]
        positive_flux_difference = np.maximum(flux_difference, 0)
        try:
            x_flux_difference_centroid = np.average(pixel_side_indexes, weights=np.mean(positive_flux_difference, axis=0))
            y_flux_difference_centroid = np.average(pixel_side_indexes, weights=np.mean(positive_flux_difference, axis=1))
            # plt.plot(target_pixel_file.column + x_flux_difference_centroid, target_pixel_file.row + y_flux_difference_centroid, 'mo')
            # x_flux_difference_centroid = np.average(pixel_side_indexes,
            #                                         weights=np.mean(flux_difference, axis=0))
            # y_flux_difference_centroid = np.average(pixel_side_indexes,
            #                                         weights=np.mean(flux_difference, axis=1))
            # plt.plot(target_pixel_file.column + x_flux_difference_centroid,
            #          target_pixel_file.row + y_flux_difference_centroid, 'co')
            # sky_coord = SkyCoord(ra=target_pixel_file.ra, dec=target_pixel_file.dec, unit=units.deg)
            # pixel_position = target_pixel_file.wcs.world_to_pixel(sky_coord)
            # plt.plot(pixel_position[0] + target_pixel_file.column, pixel_position[1] + target_pixel_file.row, 'ro')
            # plt.show()
            centroid_sky_coord = target_pixel_file.wcs.pixel_to_world(x_flux_difference_centroid,
                                                                      y_flux_difference_centroid)
            return centroid_sky_coord
        except ZeroDivisionError as error:
            raise CentroidAlgorithmFailedError from error
