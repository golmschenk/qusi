"""
Code for a class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
"""
import math
import re
import shutil
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations, Catalogs
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError
from astroquery.vizier import Vizier
from retrying import retry

from ramjet.analysis.lightcurve_visualizer import plot_lightcurve
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface, ToiColumns


class TessFluxType(Enum):
    SAP = 'SAP_FLUX'
    PDCSAP = 'PDCSAP_FLUX'


def is_common_mast_connection_error(exception: Exception) -> bool:
    """
    Returns if the passed exception is a common MAST connection error. Made for deciding whether to retry a function.

    :param exception: The exception to check.
    :return: A boolean stating if the exception is a common MAST connection error.
    """
    return (isinstance(exception, AstroQueryTimeoutError) or
            isinstance(exception, requests.exceptions.ReadTimeout) or
            isinstance(exception, requests.exceptions.ConnectionError))


class TessDataInterface:
    """
    A class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
    """
    def __init__(self):
        Observations.TIMEOUT = 2000
        Observations.PAGESIZE = 3000
        Catalogs.TIMEOUT = 2000
        Catalogs.PAGESIZE = 3000
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
    @retry(retry_on_exception=is_common_mast_connection_error)
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
    @retry(retry_on_exception=is_common_mast_connection_error)
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
    @retry(retry_on_exception=is_common_mast_connection_error)
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
        data_frame['TIC ID'] = data_frame['obs_id'].map(self.get_tic_id_from_single_sector_obs_id)
        return data_frame

    def add_sector_column_to_single_sector_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column with the sector the data was taken from.

        :param observations: The table of single-sector observations.
        :return: The table with the added sector column.
        """
        observations['Sector'] = observations['obs_id'].map(self.get_sector_from_single_sector_obs_id)
        return observations

    @staticmethod
    def load_fluxes_and_times_from_fits_file(example_path: Union[str, Path],
                                             flux_type: TessFluxType = TessFluxType.PDCSAP) -> (np.ndarray, np.ndarray):
        """
        Extract the flux and time values from a TESS FITS file.

        :param example_path: The path to the FITS file.
        :param flux_type: The flux type to extract from the FITS file.
        :return: The flux and times values from the FITS file.
        """
        with fits.open(example_path) as hdu_list:
            lightcurve = hdu_list[1].data  # Lightcurve information is in first extension table.
        fluxes = lightcurve[flux_type.value]
        times = lightcurve['TIME']
        assert times.shape == fluxes.shape
        # noinspection PyUnresolvedReferences
        nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
        fluxes = np.delete(fluxes, nan_indexes)
        times = np.delete(times, nan_indexes)
        return fluxes, times

    def download_lightcurve(self, tic_id: int, sector: int = None, save_directory: Union[Path, str] = None) -> Path:
        """
        Downloads a lightcurve from MAST.

        :param tic_id: The TIC ID of the lightcurve target to download.
        :param sector: The sector to download. If not specified, downloads first available sector.
        :param save_directory: The directory to save the FITS file to. If not specified, uses the system temporary
                               directory.
        :return: The path to the downloaded file.
        """
        observations = self.get_all_tess_time_series_observations(tic_id=tic_id)
        single_sector_observations = self.filter_for_single_sector_observations(observations)
        observations_with_sectors = self.add_sector_column_to_single_sector_observations(single_sector_observations)
        if sector is not None:
            observations_with_sectors = observations_with_sectors[observations_with_sectors['Sector'] == sector]
        else:
            observations_with_sectors = observations_with_sectors.head(1)
        product_list = self.get_product_list(observations_with_sectors)
        lightcurves_product_list = product_list[product_list['productSubGroupDescription'] == 'LC']
        manifest = self.download_products(lightcurves_product_list, data_directory=tempfile.gettempdir())
        lightcurve_path = Path(manifest['Local Path'].iloc[0])
        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            new_lightcurve_path = str(save_directory.joinpath(lightcurve_path.name))
            shutil.move(str(lightcurve_path), new_lightcurve_path)
            lightcurve_path = new_lightcurve_path
        return lightcurve_path

    def plot_lightcurve_from_mast(self, tic_id: int, sector: int = None, exclude_flux_outliers: bool = False,
                                  base_data_point_size=3):
        """
        Downloads and plots a lightcurve from MAST.

        :param tic_id: The TIC ID of the lightcurve target to download.
        :param sector: The sector to download. If not specified, downloads first available sector.
        :param exclude_flux_outliers: Whether or not to exclude flux outlier data points when plotting.
        :param base_data_point_size: The size of the data points to use when plotting (and related sizes).
        """
        lightcurve_path = self.download_lightcurve(tic_id, sector)
        fluxes, times = self.load_fluxes_and_times_from_fits_file(lightcurve_path)
        if sector is None:
            sector = self.get_sector_from_single_sector_obs_id(str(lightcurve_path.stem))
        title = f'TIC {tic_id} sector {sector}'
        if exclude_flux_outliers:
            title += ' (outliers removed)'
        plot_lightcurve(times=times, fluxes=fluxes, exclude_flux_outliers=exclude_flux_outliers, title=title,
                        base_data_point_size=base_data_point_size)

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

    def retrieve_exofop_planet_disposition_for_tic_id(self, tic_id: int) -> pd.DataFrame:
        """
        Retrieves the ExoFOP disposition information for a given TIC ID from
        <https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv>`_.

        :param tic_id: The TIC ID to get available data for.
        :return: The disposition data frame.
        """
        tess_toi_data_interface = TessToiDataInterface()
        tess_toi_data_interface.update_toi_dispositions_file()
        dispositions = tess_toi_data_interface.dispositions
        tic_target_dispositions = dispositions[dispositions['TIC ID'] == tic_id]
        return tic_target_dispositions

    def print_exofop_planet_dispositions_for_tic_target(self, tic_id):
        """
        Prints all ExoFOP disposition information for a given TESS target.

        :param tic_id: The TIC target to for.
        """
        dispositions_data_frame = self.retrieve_exofop_planet_disposition_for_tic_id(tic_id)
        if dispositions_data_frame.shape[0] == 0:
            print('No known ExoFOP dispositions found.')
            return
        # Use context options to not truncate printed data.
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print(dispositions_data_frame)

    def download_all_two_minute_cadence_lightcurves(self, save_directory: Path):
        """
        Downloads all two minute cadence lightcurves from TESS.

        :param save_directory: The directory to save the lightcurves to.
        """
        print(f'Starting download of all 2-minute cadence lightcurves to directory `{save_directory}`.')
        if save_directory.exists():
            print('Will delete existing existing directory in 10 seconds. Control-C to cancel.')
            time.sleep(10)
            print('Deleting existing data...')
            shutil.rmtree(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        print(f'Retrieving observations list from MAST...')
        tess_observations = self.get_all_tess_time_series_observations()
        single_sector_observations = self.filter_for_single_sector_observations(tess_observations)
        print(f'Retrieving data products list from MAST...')
        data_products = self.get_product_list(single_sector_observations)
        print(f'Downloading lightcurves...')
        lightcurve_data_products = data_products[data_products['productFilename'].str.endswith('lc.fits')]
        download_manifest = self.download_products(lightcurve_data_products, data_directory=save_directory)
        print(f'Moving lightcurves to {save_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
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
        single_sector_observations = self.add_sector_column_to_single_sector_observations(single_sector_observations)
        return sorted(single_sector_observations['Sector'].unique())

    def download_exofop_toi_lightcurves_to_directory(self, directory: Union[Path, str] = None):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_ lightcurve files to the
        given directory.

        :param directory: The directory to download the lightcurves to. Defaults to the data interface directory.
        """
        print("Downloading ExoFOP TOI disposition CSV...")
        tess_toi_data_interface = TessToiDataInterface()
        if directory is None:
            directory = tess_toi_data_interface.lightcurves_directory
        if isinstance(directory, str):
            directory = Path(directory)
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with tess_toi_data_interface.dispositions_path.open('wb') as csv_file:
            csv_file.write(response.content)
        toi_dispositions = tess_toi_data_interface.load_toi_dispositions_in_project_format()
        tic_ids = toi_dispositions[ToiColumns.tic_id.value].unique()
        print('Downloading TESS observation list...')
        tess_observations = self.get_all_tess_time_series_observations(tic_id=tic_ids)
        single_sector_observations = self.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = self.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_observations = self.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        print("Downloading lightcurves which are confirmed or suspected planets in TOI dispositions...")
        suspected_planet_dispositions = toi_dispositions[toi_dispositions[ToiColumns.disposition.value] != 'FP']
        suspected_planet_observations = pd.merge(single_sector_observations, suspected_planet_dispositions, how='inner',
                                                 on=[ToiColumns.tic_id.value, ToiColumns.sector.value])
        observations_not_found = suspected_planet_dispositions.shape[0] - suspected_planet_observations.shape[0]
        print(f"{suspected_planet_observations.shape[0]} observations found that match the TOI dispositions.")
        print(f"No observations found for {observations_not_found} entries in TOI dispositions.")
        suspected_planet_data_products = self.get_product_list(suspected_planet_observations)
        suspected_planet_lightcurve_data_products = suspected_planet_data_products[
            suspected_planet_data_products['productFilename'].str.endswith('lc.fits')
        ]
        suspected_planet_download_manifest = self.download_products(
            suspected_planet_lightcurve_data_products, data_directory=tess_toi_data_interface.data_directory)
        print(f'Moving lightcurves to {directory}...')
        directory.mkdir(parents=True, exist_ok=True)
        for file_path_string in suspected_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            lightcurve_path = directory.joinpath(file_path.name)
            file_path.rename(lightcurve_path)
            try:
                hdu_list = fits.open(str(lightcurve_path))
                lightcurve = hdu_list[1].data
                _ = lightcurve['TIME'][0]
            except (OSError, TypeError):
                print(f'{file_path} seems to be corrupt. Re-downloading and replacing.')
                sector = tess_data_interface.get_sector_from_single_sector_obs_id(str(lightcurve_path.stem))
                tic_id = tess_data_interface.get_tic_id_from_single_sector_obs_id(str(lightcurve_path.stem))
                tess_data_interface.download_lightcurve(tic_id, sector, save_directory=lightcurve_path.parent)

    @staticmethod
    def get_tic_id_and_sector_from_file_path(file_path: Union[Path, str]):
        """
        Add general purpose function to get the TIC ID and sector from commonly encountered file name patterns.

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
    tess_data_interface.download_all_two_minute_cadence_lightcurves(Path('data/tess_two_minute_cadence_lightcurves'))
