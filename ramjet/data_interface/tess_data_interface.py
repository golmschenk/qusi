"""
Code for a class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
"""
import astroquery
import lightkurve

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import math
import re
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations, Catalogs
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError
from astroquery.vizier import Vizier
from bokeh.io import show
from retrying import retry
from bokeh.plotting import Figure

from ramjet.analysis.light_curve_visualizer import plot_light_curve, create_dual_light_curve_figure


class TessFluxType(Enum):
    """
    An enum to represent the types of available fluxes in TESS two minute data.
    """
    SAP = 'SAP_FLUX'
    PDCSAP = 'PDCSAP_FLUX'


class ColumnName(StrEnum):
    """
    An enum for the names of the columns produced by the data interface class.
    """
    TIC_ID = 'TIC ID'
    SECTOR = 'Sector'


def is_common_mast_connection_error(exception: Exception) -> bool:
    """
    Returns if the passed exception is a common MAST connection error. Made for deciding whether to retry a function.

    :param exception: The exception to check.
    :return: A boolean stating if the exception is a common MAST connection error.
    """
    print(f'Retrying on {exception}...', flush=True)
    # TODO: Rename function, as it includes more than just MAST now.
    return (isinstance(exception, AstroQueryTimeoutError) or
            isinstance(exception, TimeoutError) or
            isinstance(exception, requests.exceptions.ReadTimeout) or
            isinstance(exception, requests.exceptions.ChunkedEncodingError) or
            isinstance(exception, requests.exceptions.HTTPError) or
            isinstance(exception, requests.exceptions.ConnectionError) or
            isinstance(exception, ConnectionResetError) or
            isinstance(exception, lightkurve.search.SearchError) or
            isinstance(exception, astroquery.exceptions.RemoteServiceError))


class NoDataProductsFoundException(Exception):
    """An exception when no data products are found from MAST."""
    pass


class TessDataInterface:
    """
    A class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
    """
    def __init__(self):
        Observations.TIMEOUT = 2000
        Observations.PAGESIZE = 3000
        Catalogs.TIMEOUT = 2000
        Catalogs.PAGESIZE = 3000
        try:  # Temporary fix for astroquery's update of timeout and pagesize locations.
            Observations._portal_api_connection.TIMEOUT = 2000
            Observations._portal_api_connection.PAGESIZE = 3000
            Catalogs._portal_api_connection.TIMEOUT = 2000
            Catalogs._portal_api_connection.PAGESIZE = 3000
        except AttributeError:
            pass
        self.mast_input_query_chunk_size = 1000

    def get_all_tess_time_series_observations(self, tic_id: Union[int, List[int]] = None) -> pd.DataFrame:
        """
        Gets all TESS time-series observations, limited to science data product level. Breaks large queries up to make
        the communication with MAST smoother.

        :param tic_id: An optional TIC ID or list of TIC IDs to limit the query to.
        :return: The list of time series observations as rows in a Pandas data frame.
        """
        if tic_id is None or np.isscalar(tic_id):
            observations = self.get_all_tess_time_series_observations_chunk(tic_id)
        else:
            observations = None
            for tic_id_list_chunk in np.array_split(tic_id, math.ceil(len(tic_id) / self.mast_input_query_chunk_size)):
                observations_chunk = self.get_all_tess_time_series_observations_chunk(tic_id_list_chunk)
                if observations is None:
                    observations = observations_chunk
                else:
                    observations = observations.append(observations_chunk, ignore_index=True)
        return observations

    @staticmethod
    @retry(retry_on_exception=is_common_mast_connection_error, stop_max_attempt_number=10)
    def get_all_tess_time_series_observations_chunk(tic_id: Union[int, List[int]] = None) -> pd.DataFrame:
        """
        Gets all TESS time-series observations, limited to science data product level. Repeats download attempt on
        error.

        :param tic_id: An optional TIC ID or list of TIC IDs to limit the query to.
        :return: The list of time series observations as rows in a Pandas data frame.
        """
        if tic_id is None:
            tic_id = []  # When the empty list is passed to `query_criteria`, any value is considered a match.
        tess_observations = Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries',
                                                        calib_level=3,  # Science data product level.
                                                        target_name=tic_id)
        return tess_observations.to_pandas()

    def get_product_list(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        A wrapper for MAST's `get_product_list`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Breaks large queries up to make the communication with MAST smoother.

        :param observations: The data frame of observations to get. Will be converted from DataFrame to Table for query.
        :return: The data frame of the product list. Will be converted from Table to DataFrame for use.
        """
        if observations.shape[0] > 1:
            product_list = None
            for observations_chunk in np.array_split(observations,
                                                     math.ceil(observations.shape[0] / self.mast_input_query_chunk_size)):
                product_list_chunk = self.get_product_list_chunk(observations_chunk)
                if product_list is None:
                    product_list = product_list_chunk
                else:
                    product_list = product_list.append(product_list_chunk, ignore_index=True)
        else:
            product_list = self.get_product_list_chunk(observations)
        return product_list

    @staticmethod
    @retry(retry_on_exception=is_common_mast_connection_error, stop_max_attempt_number=10)
    def get_product_list_chunk(observations: pd.DataFrame) -> pd.DataFrame:
        """
        A wrapper for MAST's `get_product_list`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Retries on error when communicating with the MAST server.

        :param observations: The data frame of observations to get. Will be converted from DataFrame to Table for query.
        :return: The data frame of the product list. Will be converted from Table to DataFrame for use.
        """
        data_products = Observations.get_product_list(Table.from_pandas(observations))
        return data_products.to_pandas()

    @staticmethod
    @retry(retry_on_exception=is_common_mast_connection_error, stop_max_attempt_number=10)
    def download_products(data_products: pd.DataFrame, data_directory: Path) -> pd.DataFrame:
        """
         A wrapper for MAST's `download_products`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Retries on error when communicating with the MAST server.

        :param data_products: The data frame of data products to download. Will be converted from DataFrame to Table
                              for sending the request to MAST.
        :param data_directory: The path to download the data to.
        :return: The manifest of the download. Will be converted from Table to DataFrame for use.
        """
        manifest = Observations.download_products(Table.from_pandas(data_products), download_dir=str(data_directory))
        if manifest is None:
            raise NoDataProductsFoundException
        return manifest.to_pandas()

    @staticmethod
    def filter_for_single_sector_observations(time_series_observations: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a data frame of observations to get only the single sector observations.

        :param time_series_observations: A data frame of observations to filter for single sector observations.
        :return: The data frame of single sector observations.
        """
        single_sector_observations = time_series_observations[
            time_series_observations['dataURL'].str.endswith('lc.fits')
        ]
        return single_sector_observations.copy()

    @staticmethod
    def filter_out_twenty_second_cadence_observations(time_series_observations: pd.DataFrame) -> pd.DataFrame:
        """
        Removes 20-second cadence data from the observation data frame.

        :param time_series_observations: A data frame of observations to filtered.
        :return: The data frame without 20-second cadence data.
        """
        observations_without_twenty_second_cadence = time_series_observations[
            ~time_series_observations['dataURL'].str.endswith('fast-lc.fits')
        ]
        return observations_without_twenty_second_cadence.copy()

    @staticmethod
    def filter_for_multi_sector_observations(time_series_observations: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a data frame of observations to get only the multi sector observations.

        :param time_series_observations: A data frame of observations to filter for multi sector observations.
        :return: The data frame of multi sector observations.
        """
        multi_sector_observations = time_series_observations[
            time_series_observations['dataURL'].str.endswith('dvt.fits')
        ]
        return multi_sector_observations.copy()

    @staticmethod
    def get_tic_id_from_single_sector_obs_id(obs_id: str) -> int:
        """
        Extracts the TIC ID from a single-sector obs_id string.

        :param obs_id: The obs_id to extract from.
        :return: The extracted TIC ID.
        """
        return int(obs_id.split('-')[2].lstrip('0'))

    @staticmethod
    def get_sector_from_single_sector_obs_id(obs_id: str) -> int:
        """
        Extracts the sector from a single-sector obs_id string.

        :param obs_id: The obs_id to extract from.
        :return: The extracted sector number.
        """
        return int(obs_id.split('-')[1][1:])

    def add_tic_id_column_to_single_sector_observations(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column with the TIC ID the row is related to.

        :param data_frame: The data frame of single-sector entries.
        :return: The table with the added TIC ID column.
        """
        data_frame[ColumnName.TIC_ID] = data_frame['obs_id'].map(self.get_tic_id_from_single_sector_obs_id)
        return data_frame

    def add_sector_column_to_single_sector_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column with the sector the data was taken from.

        :param observations: The table of single-sector observations.
        :return: The table with the added sector column.
        """
        observations[ColumnName.SECTOR] = observations['obs_id'].map(self.get_sector_from_single_sector_obs_id)
        return observations

    def load_light_curve_from_fits_file(self, light_curve_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Loads a light_curve from a FITS file in a dictionary form with the structure of the FITS arrays.

        :param light_curve_path: The path to the FITS file.
        :return: The light_curve.
        """
        try:
            with fits.open(light_curve_path) as hdu_list:
                light_curve = hdu_list[1].data  # Light curve information is in first extension table.
        except OSError:  # If the FITS file is corrupt, re-download (seems to happen often enough).
            light_curve_path = Path(light_curve_path)  # In case it's currently a string.
            light_curve_path.unlink()
            tic_id, sector = self.get_tic_id_and_sector_from_file_path(light_curve_path)
            self.download_two_minute_cadence_light_curve(tic_id=tic_id, sector=sector,
                                                         save_directory=light_curve_path.parent)
            with fits.open(light_curve_path) as hdu_list:
                light_curve = hdu_list[1].data  # Light curve information is in first extension table.
        return light_curve

    def load_fluxes_and_times_from_fits_file(self, light_curve_path: Union[str, Path],
                                             flux_type: TessFluxType = TessFluxType.PDCSAP,
                                             remove_nans: bool = True) -> (np.ndarray, np.ndarray):
        """
        Extract the flux and time values from a TESS FITS file.

        :param light_curve_path: The path to the FITS file.
        :param flux_type: The flux type to extract from the FITS file.
        :param remove_nans: Whether or not to remove nans.
        :return: The flux and times values from the FITS file.
        """
        light_curve = self.load_light_curve_from_fits_file(light_curve_path)
        fluxes = light_curve[flux_type.value]
        times = light_curve['TIME']
        assert times.shape == fluxes.shape
        if remove_nans:
            # noinspection PyUnresolvedReferences
            nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
            fluxes = np.delete(fluxes, nan_indexes)
            times = np.delete(times, nan_indexes)
        return fluxes, times

    def load_fluxes_flux_errors_and_times_from_fits_file(self, light_curve_path: Union[str, Path],
                                                         flux_type: TessFluxType = TessFluxType.PDCSAP,
                                                         remove_nans: bool = True
                                                         ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Extract the flux and time values from a TESS FITS file.

        :param light_curve_path: The path to the FITS file.
        :param flux_type: The flux type to extract from the FITS file.
        :param remove_nans: Whether or not to remove nans.
        :return: The flux and times values from the FITS file.
        """
        light_curve = self.load_light_curve_from_fits_file(light_curve_path)
        fluxes = light_curve[flux_type.value]
        flux_errors = light_curve[flux_type.value + '_ERR']
        times = light_curve['TIME']
        assert times.shape == fluxes.shape
        if remove_nans:
            # noinspection PyUnresolvedReferences
            nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.union1d(np.argwhere(np.isnan(times)),
                                                                               np.argwhere(np.isnan(flux_errors))))
            fluxes = np.delete(fluxes, nan_indexes)
            flux_errors = np.delete(flux_errors, nan_indexes)
            times = np.delete(times, nan_indexes)
        return fluxes, flux_errors, times

    def download_two_minute_cadence_light_curve(self, tic_id: int, sector: int = None,
                                               save_directory: Union[Path, str] = None) -> Path:
        """
        Downloads a light curve from MAST.

        :param tic_id: The TIC ID of the light curve target to download.
        :param sector: The sector to download. If not specified, downloads first available sector.
        :param save_directory: The directory to save the FITS file to. If not specified, uses the system temporary
                               directory.
        :return: The path to the downloaded file.
        """
        observations = self.get_all_tess_time_series_observations(tic_id=tic_id)
        single_sector_observations = self.filter_for_single_sector_observations(observations)
        two_minute_observations = self.filter_out_twenty_second_cadence_observations(single_sector_observations)
        observations_with_sectors = self.add_sector_column_to_single_sector_observations(two_minute_observations)
        if sector is not None:
            observations_with_sectors = observations_with_sectors[
                observations_with_sectors[ColumnName.SECTOR] == sector]
        else:
            observations_with_sectors = observations_with_sectors.head(1)
        product_list = self.get_product_list(observations_with_sectors)
        light_curves_product_list = product_list[product_list['productSubGroupDescription'] == 'LC']
        manifest = self.download_products(light_curves_product_list, data_directory=Path(tempfile.gettempdir()))
        light_curve_path = Path(manifest['Local Path'].iloc[0])
        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            new_light_curve_path = save_directory.joinpath(light_curve_path.name)
            shutil.move(str(light_curve_path), str(new_light_curve_path))
            light_curve_path = new_light_curve_path
        return light_curve_path

    def plot_light_curve_from_mast(self, tic_id: int, sector: int = None, exclude_flux_outliers: bool = False,
                                  base_data_point_size=3):
        """
        Downloads and plots a light curve from MAST.

        :param tic_id: The TIC ID of the light curve target to download.
        :param sector: The sector to download. If not specified, downloads first available sector.
        :param exclude_flux_outliers: Whether or not to exclude flux outlier data points when plotting.
        :param base_data_point_size: The size of the data points to use when plotting (and related sizes).
        """
        light_curve_path = self.download_two_minute_cadence_light_curve(tic_id, sector)
        fluxes, times = self.load_fluxes_and_times_from_fits_file(light_curve_path)
        if sector is None:
            sector = self.get_sector_from_single_sector_obs_id(str(light_curve_path.stem))
        title = f'TIC {tic_id} sector {sector}'
        if exclude_flux_outliers:
            title += ' (outliers removed)'
        plot_light_curve(times=times, fluxes=fluxes, title=title, exclude_flux_outliers=exclude_flux_outliers,
                         base_data_point_size=base_data_point_size)

    def create_pdcsap_and_sap_comparison_figure_from_mast(self, tic_id: int, sector: int = None) -> Figure:
        """
        Creates a comparison figure containing both the PDCSAP and SAP signals.

        :param tic_id: The TIC ID of the light curve to plot.
        :param sector: The sector of the light curve to plot.
        :return: The generated figure.
        """
        light_curve_path = self.download_two_minute_cadence_light_curve(tic_id, sector)
        pdcsap_fluxes, pdcsap_times = self.load_fluxes_and_times_from_fits_file(light_curve_path, TessFluxType.PDCSAP)
        normalized_pdcsap_fluxes = pdcsap_fluxes / np.median(pdcsap_fluxes)
        sap_fluxes, sap_times = self.load_fluxes_and_times_from_fits_file(light_curve_path, TessFluxType.SAP)
        normalized_sap_fluxes = sap_fluxes / np.median(sap_fluxes)
        if sector is None:
            _, sector = self.get_tic_id_and_sector_from_file_path(light_curve_path)
        title = f'TIC {tic_id} sector {sector}'
        figure = create_dual_light_curve_figure(fluxes0=normalized_pdcsap_fluxes, times0=pdcsap_times, name0='PDCSAP',
                                                fluxes1=normalized_sap_fluxes, times1=sap_times, name1='SAP',
                                                title=title, x_axis_label='Time (BTJD)')
        return figure

    def show_pdcsap_and_sap_comparison_from_mast(self, tic_id: int, sector: int = None):
        """
        Shows a comparison figure containing both the PDCSAP and SAP signals.

        :param tic_id: The TIC ID of the light curve to plot.
        :param sector: The sector of the light curve to plot.
        """
        comparison_figure = self.create_pdcsap_and_sap_comparison_figure_from_mast(tic_id, sector)
        comparison_figure.sizing_mode = 'stretch_width'
        show(comparison_figure)

    def show_light_curve(self, light_curve_path: Path):
        """
        Shows a figure of the light curve at the passed path.

        :param light_curve_path: The path of the light curve.
        """
        fluxes, times = self.load_fluxes_and_times_from_fits_file(light_curve_path)
        figure = Figure(title=str(light_curve_path), x_axis_label='Flux', y_axis_label='Time', active_drag='box_zoom')
        color = 'mediumblue'
        figure.line(times, fluxes, line_color=color, line_alpha=0.1)
        figure.circle(times, fluxes, line_color=color, line_alpha=0.4, fill_color=color, fill_alpha=0.1)
        figure.sizing_mode = 'stretch_width'
        show(figure)

    @staticmethod
    def get_target_coordinates(tic_id: int) -> SkyCoord:
        """
        Get the sky coordinates of the target by a TIC ID.

        :param tic_id: The target's TIC ID.
        :return: The coordinates of the target.
        """
        target_observations = Catalogs.query_criteria(catalog='TIC', ID=tic_id).to_pandas()
        ra = target_observations['ra'].iloc[0]
        dec = target_observations['dec'].iloc[0]
        return SkyCoord(ra, dec, unit='deg')

    @staticmethod
    @retry(retry_on_exception=is_common_mast_connection_error, stop_max_attempt_number=10)
    def get_tess_input_catalog_row(tic_id: int) -> pd.Series:
        """
        Get the TIC row for a TIC ID.

        :param tic_id: The target's TIC ID.
        :return: The row of a the TIC corresponding to the TIC ID.
        """
        target_observations = Catalogs.query_criteria(catalog='TIC', ID=tic_id).to_pandas()
        return target_observations.iloc[0]

    @staticmethod
    def get_variable_data_frame_for_coordinates(coordinates, radius='21s') -> pd.DataFrame:
        """
        Gets a data frame containing all known variables within a radius of the given coordinates.

        :param coordinates: The coordinates to search.
        :param radius: The radius to search. TESS has a pixel size of 21 arcseconds across.
        :return: The data frame of the variables. Returns an empty data frame if none exist.
        """
        variable_table_list = Vizier.query_region(coordinates, radius=radius, catalog='B/gcvs/gcvs_cat')
        if len(variable_table_list) > 0:
            return variable_table_list[0].to_pandas()
        else:
            return pd.DataFrame()

    def get_variable_data_frame_for_tic_id(self, tic_id):
        """
        Gets a data frame containing all known variables near a TIC target (including the target if applicable).

        :param tic_id: The TIC target to search for variables near.
        :return: A data frame of the variables near the target (including the target if applicable).
        """
        coordinates = self.get_target_coordinates(tic_id)
        return self.get_variable_data_frame_for_coordinates(coordinates)

    def print_variables_near_tess_target(self, tic_id):
        """
        Prints all variable stars near a given TESS target (including the target if applicable).

        :param tic_id: The TIC target to search for variables near.
        """
        variable_data_frame = self.get_variable_data_frame_for_tic_id(tic_id)
        if variable_data_frame.shape[0] == 0:
            print('No known variables found.')
            return
        print('Variable type abbreviation explanations: http://www.sai.msu.su/gcvs/gcvs/vartype.htm')
        print_data_frame = pd.DataFrame()
        print_data_frame['Variable Type'] = variable_data_frame['VarType'].str.decode('utf-8')
        print_data_frame['Max magnitude'] = variable_data_frame['magMax']
        print_data_frame['Period (days)'] = variable_data_frame['Period']
        print_data_frame.sort_values('Max magnitude', inplace=True)
        print_data_frame.reset_index(drop=True, inplace=True)
        print(print_data_frame)

    def download_two_minute_cadence_light_curves(self, save_directory: Path, limit: Union[None, int] = None):
        """
        Downloads all two minute cadence light curves from TESS.

        :param save_directory: The directory to save the light curves to.
        :param limit: Limits the number of light curves downloaded. Default of None will download all light curves.
        """
        print(f'Starting download of 2-minute cadence light curves to directory `{save_directory}`.')
        save_directory.mkdir(parents=True, exist_ok=True)
        print(f'Retrieving observations list from MAST...')
        single_sector_observations = self.get_all_two_minute_single_sector_observations()
        print(f'Retrieving data products list from MAST...')
        data_products = self.get_product_list(single_sector_observations)
        print(f'Downloading light curves...')
        light_curve_data_products = data_products[data_products['productFilename'].str.endswith('lc.fits')]
        if limit is not None:
            light_curve_data_products = light_curve_data_products.sample(frac=1, random_state=0).head(limit)
        download_manifest = self.download_products(light_curve_data_products, data_directory=save_directory)
        print(f'Moving light curves to {save_directory}...')
        for _, manifest_row in download_manifest.iterrows():
            if manifest_row['Status'] == 'COMPLETE':
                file_path = Path(manifest_row['Local Path'])
                file_path.rename(save_directory.joinpath(file_path.name))
        print('Database ready.')

    def get_sectors_target_appears_in(self, tic_id: int) -> List:
        """
        Gets the list of sectors a TESS target appears in.

        :param tic_id: The TIC ID of the target.
        :return: The list of sectors.
        """
        time_series_observations = self.get_all_tess_time_series_observations(tic_id)
        single_sector_observations = self.filter_for_single_sector_observations(time_series_observations)
        two_minute_observations = self.filter_out_twenty_second_cadence_observations(single_sector_observations)
        single_sector_observations = self.add_sector_column_to_single_sector_observations(two_minute_observations)
        return sorted(single_sector_observations[ColumnName.SECTOR].unique())

    def get_all_two_minute_single_sector_observations(self, tic_ids: List[int] = None) -> pd.DataFrame:
        """
        Gets the data frame containing all the single sector observations, with TIC ID and sector columns.

        :param tic_ids: The TIC IDs to get observations for.
        :return: The data frame containing the observation information.
        """
        tess_observations = self.get_all_tess_time_series_observations(tic_id=tic_ids)
        single_sector_observations = self.filter_for_single_sector_observations(tess_observations)
        two_minute_observations = self.filter_out_twenty_second_cadence_observations(single_sector_observations)
        two_minute_observations = self.add_tic_id_column_to_single_sector_observations(
            two_minute_observations)
        two_minute_observations = self.add_sector_column_to_single_sector_observations(
            two_minute_observations)
        return two_minute_observations

    def verify_light_curve(self, light_curve_path: Path):
        """
        The light_curve is checked if it's malformed, and if it is, it is re-downloaded.

        :param light_curve_path: The path of the light_curve.
        """
        try:
            hdu_list = fits.open(str(light_curve_path))
            light_curve = hdu_list[1].data
            _ = light_curve['TIME'][0]  # Basic check if the light_curve file is malformed.
        except (OSError, TypeError):
            print(f'{light_curve_path} seems to be malformed. Re-downloading and replacing.')
            sector = self.get_sector_from_single_sector_obs_id(str(light_curve_path.stem))
            tic_id = self.get_tic_id_from_single_sector_obs_id(str(light_curve_path.stem))
            self.download_two_minute_cadence_light_curve(tic_id, sector, save_directory=light_curve_path.parent)

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


if __name__ == '__main__':
    tess_data_interface = TessDataInterface()
    if len(sys.argv) > 1:
        limit_ = int(sys.argv[1])
    else:
        limit_ = None
    tess_data_interface.download_two_minute_cadence_light_curves(Path('data/tess_two_minute_cadence_light_curves'),
                                                                 limit=limit_)
