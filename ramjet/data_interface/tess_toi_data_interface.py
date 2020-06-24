from enum import Enum
from pathlib import Path
from typing import Union

import pandas as pd
import requests

class ToiColumns(Enum):
    """
    An enum for the names of the TOI columns for Pandas data frames.
    """
    tic_id = 'TIC ID'
    disposition = 'Disposition'
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
        columns_to_use = ['TIC ID', 'TFOPWG Disposition', 'Epoch (BJD)', 'Period (days)', 'Duration (hours)', 'Sectors']
        dispositions = pd.read_csv(self.dispositions_path, usecols=columns_to_use)
        dispositions.rename(columns={'TFOPWG Disposition': ToiColumns.disposition.value,
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
