"""
Code to represent the database of TESS data based on `Liang Yu's work <https://arxiv.org/pdf/1904.02726.pdf>`_.
"""

import pandas as pd
import requests
from pathlib import Path

from photometric_database.tess_transit_lightcurve_label_per_time_step_database import \
    TessTransitLightcurveLabelPerTimeStepDatabase


class LiangYuLightcurveDatabase(TessTransitLightcurveLabelPerTimeStepDatabase):
    """
    A class to represent the database of TESS data based on `Liang Yu's work <https://arxiv.org/pdf/1904.02726.pdf>`_.
    """

    def __init__(self, data_directory='data/tess'):
        super().__init__(data_directory=data_directory)
        self.liang_yu_dispositions_path = self.data_directory.joinpath('liang_yu_dispositions.csv')

    def get_meta_data_frame_for_available_lightcurves(self) -> pd.DataFrame:
        """
        Gets the available meta disposition data from Liang Yu's work and combines it with the available lightcurve
        data, throwing out any data that doesn't have its counterpart.

        :return: The meta data frame containing the lightcurve paths and meta data needed to generate labels.
        """
        # noinspection SpellCheckingInspection
        columns_to_use = ['tic_id', 'Disposition', 'Epoc', 'Period', 'Duration', 'Sectors']
        liang_yu_dispositions = pd.read_csv(self.liang_yu_dispositions_path, usecols=columns_to_use)
        # noinspection SpellCheckingInspection
        liang_yu_dispositions.rename(columns={'Disposition': 'disposition', 'Epoc': 'transit_epoch',
                                              'Period': 'transit_period', 'Duration': 'transit_duration',
                                              'Sectors': 'sector'}, inplace=True)
        lightcurve_paths = list(self.lightcurve_directory.glob('*lc.fits'))
        tic_ids = [int(self.get_tic_id_from_single_sector_obs_id(path.name)) for path in lightcurve_paths]
        sectors = [self.get_sector_from_single_sector_obs_id(path.name) for path in lightcurve_paths]
        lightcurve_meta_data = pd.DataFrame({'lightcurve_path': lightcurve_paths, 'tic_id': tic_ids, 'sector': sectors})
        meta_data = pd.merge(liang_yu_dispositions, lightcurve_meta_data, how='inner', on=['tic_id', 'sector'])
        return meta_data

    def download_liang_yu_database(self):
        """
        Downloads the database used by https://arxiv.org/pdf/1904.02726.pdf.
        """
        print("Downloading Liang Yu's disposition CSV...")
        liang_yu_csv_url = 'https://raw.githubusercontent.com/yuliang419/Astronet-Triage/master/astronet/tces.csv'
        response = requests.get(liang_yu_csv_url)
        with open(self.liang_yu_dispositions_path, 'wb') as csv_file:
            csv_file.write(response.content)
        print('Downloading TESS observation list...')
        tess_observations = self.get_all_tess_time_series_observations()
        single_sector_observations = self.get_single_sector_observations(tess_observations)
        single_sector_observations = self.add_sector_column_based_on_single_sector_obs_id(single_sector_observations)
        single_sector_observations['tic_id'] = single_sector_observations['target_name'].astype(int)
        print("Downloading lightcurves which appear in Liang Yu's disposition...")
        # noinspection SpellCheckingInspection
        columns_to_use = ['tic_id', 'Disposition', 'Epoc', 'Period', 'Duration', 'Sectors']
        liang_yu_dispositions = pd.read_csv(self.liang_yu_dispositions_path, usecols=columns_to_use)
        liang_yu_observations = pd.merge(single_sector_observations, liang_yu_dispositions, how='inner',
                                         left_on=['tic_id', 'sector'], right_on=['tic_id', 'Sectors'])
        number_of_observations_not_found = liang_yu_dispositions.shape[0] - liang_yu_observations.shape[0]
        print(f"{liang_yu_observations.shape[0]} observations found that match Liang Yu's entries.")
        print(f'Liang Yu used the FFIs, not the lightcurve products, so many will be missing.')
        print(f"No observations found for {number_of_observations_not_found} entries in Liang Yu's disposition.")
        liang_yu_data_products = self.get_data_products(liang_yu_observations)
        liang_yu_lightcurve_data_products = liang_yu_data_products[
            liang_yu_data_products['productFilename'].str.endswith('lc.fits')
        ]
        download_manifest = self.download_products(liang_yu_lightcurve_data_products)
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')


if __name__ == '__main__':
    LiangYuLightcurveDatabase().download_liang_yu_database()
