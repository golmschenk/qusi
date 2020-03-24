from enum import Enum
from pathlib import Path
from typing import Union

import pandas as pd
import requests

from ramjet.data_interface.tess_data_interface import TessDataInterface


class ToiColumns(Enum):
    """
    An enum for the names of the TOI columns for Pandas data frames.
    """
    tic_id = 'TIC ID'
    disposition = 'Disposition'
    planet_number = 'Planet number'
    transit_epoch__bjd = 'Transit epoch (BJD)'
    transit_period__days = 'Transit period (days)'
    transit_duration = 'Transit duration (hours)'
    sector = 'Sector'


class TessToiDataInterface:
    """
    A data interface for working with the TESS table of objects of interest.
    """
    dispositions_: pd.DataFrame = None

    def __init__(self, data_directory='data/tess_toi'):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')
        self.lightcurves_directory = self.data_directory.joinpath('lightcurves')

    @property
    def dispositions(self):
        """
        The TOI dispositions data frame property. Will load as a single class attribute on first access. If the
        data file does not exists, downloads it first.

        :return: The TOI dispositions data frame.
        """
        if self.dispositions_ is None:
            if not self.dispositions_path.exists():
                toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
                response = requests.get(toi_csv_url)
                with self.dispositions_path.open('wb') as csv_file:
                    csv_file.write(response.content)
            self.dispositions_ = self.load_toi_dispositions_in_project_format()
        return self.dispositions_

    def update_toi_dispositions_file(self):
        """
        Downloads the latest TOI dispositions file.
        """
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with self.dispositions_path.open('wb') as csv_file:
            csv_file.write(response.content)
        self.dispositions_ = self.load_toi_dispositions_in_project_format()

    def load_toi_dispositions_in_project_format(self) -> pd.DataFrame:
        """
        Loads the ExoFOP TOI table information from CSV to a data frame using a project consistent naming scheme.

        :return: The data frame of the TOI dispositions table.
        """
        columns_to_use = ['TIC ID', 'TFOPWG Disposition', 'Planet Num', 'Epoch (BJD)', 'Period (days)',
                          'Duration (hours)', 'Sectors']
        dispositions = pd.read_csv(self.dispositions_path, usecols=columns_to_use)
        dispositions.rename(columns={'TFOPWG Disposition': ToiColumns.disposition.value,
                                     'Planet Num': ToiColumns.planet_number.value,
                                     'Epoch (BJD)': ToiColumns.transit_epoch__bjd.value,
                                     'Period (days)': ToiColumns.transit_period__days.value,
                                     'Duration (hours)': ToiColumns.transit_duration.value,
                                     'Sectors': ToiColumns.sector.value}, inplace=True)
        dispositions[ToiColumns.disposition.value] = dispositions[ToiColumns.disposition.value].fillna('')
        dispositions = dispositions[dispositions[ToiColumns.sector.value].notna()]
        dispositions[ToiColumns.sector.value] = dispositions[ToiColumns.sector.value].str.split(',')
        dispositions = dispositions.explode(ToiColumns.sector.value)
        dispositions[ToiColumns.sector.value] = pd.to_numeric(dispositions[ToiColumns.sector.value]
                                                              ).astype(pd.Int64Dtype())
        return dispositions

    def download_exofop_toi_lightcurves_to_directory(self, directory: Union[Path, str] = None):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_ lightcurve files to the
        given directory.

        :param directory: The directory to download the lightcurves to. Defaults to the data interface directory.
        """
        print("Downloading ExoFOP TOI disposition CSV...")

        if directory is None:
            directory = self.lightcurves_directory
        if isinstance(directory, str):
            directory = Path(directory)
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with self.dispositions_path.open('wb') as csv_file:
            csv_file.write(response.content)
        toi_dispositions = self.load_toi_dispositions_in_project_format()
        tic_ids = toi_dispositions[ToiColumns.tic_id.value].unique()
        print('Downloading TESS obdservation list...')
        tess_data_interface = TessDataInterface()
        tess_observations = tess_data_interface.get_all_tess_time_series_observations(tic_id=tic_ids)
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = tess_data_interface.add_tic_id_column_to_single_sector_observations(
            single_sector_observations)
        single_sector_observations = tess_data_interface.add_sector_column_to_single_sector_observations(
            single_sector_observations)
        print("Downloading lightcurves which are confirmed or suspected planets in TOI dispositions...")
        suspected_planet_dispositions = toi_dispositions[toi_dispositions[ToiColumns.disposition.value] != 'FP']
        suspected_planet_observations = pd.merge(single_sector_observations, suspected_planet_dispositions, how='inner',
                                                 on=[ToiColumns.tic_id.value, ToiColumns.sector.value])
        observations_not_found = suspected_planet_dispositions.shape[0] - suspected_planet_observations.shape[0]
        print(f"{suspected_planet_observations.shape[0]} observations found that match the TOI dispositions.")
        print(f"No observations found for {observations_not_found} entries in TOI dispositions.")
        suspected_planet_data_products = tess_data_interface.get_product_list(suspected_planet_observations)
        suspected_planet_lightcurve_data_products = suspected_planet_data_products[
            suspected_planet_data_products['productFilename'].str.endswith('lc.fits')
        ]
        suspected_planet_download_manifest = tess_data_interface.download_products(
            suspected_planet_lightcurve_data_products, data_directory=self.data_directory)
        print(f'Moving lightcurves to {directory}...')
        for file_path_string in suspected_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(directory.joinpath(file_path.name))