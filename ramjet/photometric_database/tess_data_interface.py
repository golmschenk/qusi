"""
Code for a class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
"""
from enum import Enum
from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError


class TessFluxType(Enum):
    SAP = 'SAP_FLUX'
    PDCSAP = 'PDCSAP_FLUX'


class TessDataInterface:
    """
    A class for common interfacing with TESS data, such as downloading, sorting, and manipulating.
    """
    def __init__(self):
        Observations.TIMEOUT = 1200
        Observations.PAGESIZE = 10000

    @staticmethod
    def get_all_tess_time_series_observations(tic_id: Union[int, List[int]] = None) -> pd.DataFrame:
        """
        Gets all TESS time-series observations, limited to science data product level. Repeats download attempt on
        error.

        :param tic_id: An optional TIC ID or list of TIC IDs to limit the query to.
        :return: The list of time series observations as rows in a Pandas data frame.
        """
        if tic_id is None:
            tic_id = []  # When the empty list is passed to `query_criteria`, any value is considered a match.
        tess_observations = None
        while tess_observations is None:
            try:
                # noinspection SpellCheckingInspection
                tess_observations = Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries',
                                                                calib_level=3,  # Science data product level.
                                                                target_name=tic_id)
            except (AstroQueryTimeoutError, ConnectionError):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
        return tess_observations.to_pandas()

    @staticmethod
    def get_product_list(observations: pd.DataFrame) -> pd.DataFrame:
        """
        A wrapper for MAST's `get_product_list`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Retries on error when communicating with the MAST server.

        :param observations: The data frame of observations to get. Will be converted from DataFrame to Table for query.
        :return: The data frame of the product list. Will be converted from Table to DataFrame for use.
        """
        data_products = None
        while data_products is None:
            try:
                # noinspection SpellCheckingInspection
                data_products = Observations.get_product_list(Table.from_pandas(observations))
            except (AstroQueryTimeoutError, ConnectionError):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
        return data_products.to_pandas()

    @staticmethod
    def download_products(data_products: pd.DataFrame, data_directory: Path) -> pd.DataFrame:
        """
         A wrapper for MAST's `download_products`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Retries on error when communicating with the MAST server.

        :param data_products: The data frame of data products to download. Will be converted from DataFrame to Table
                              for sending the request to MAST.
        :param data_directory: The path to download the data to.
        :return: The manifest of the download. Will be converted from Table to DataFrame for use.
        """
        manifest = None
        while manifest is None:
            try:
                # noinspection SpellCheckingInspection
                manifest = Observations.download_products(Table.from_pandas(data_products),
                                                          download_dir=str(data_directory))
            except (AstroQueryTimeoutError, ConnectionError):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
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
                                             flux_type: TessFluxType = TessFluxType.SAP) -> (np.ndarray, np.ndarray):
        """
        Extract the flux and time values from a TESS FITS file.

        :param example_path: The path to the FITS file.
        :param flux_type: The flux type to extract from the FITS file.
        :return: The flux and times values from the FITS file.
        """
        hdu_list = fits.open(example_path)
        lightcurve = hdu_list[1].data  # Lightcurve information is in first extension table.
        fluxes = lightcurve[flux_type.value]
        times = lightcurve['TIME']
        assert times.shape == fluxes.shape
        # noinspection PyUnresolvedReferences
        nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
        fluxes = np.delete(fluxes, nan_indexes)
        times = np.delete(times, nan_indexes)
        return fluxes, times
