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
    def download_liang_yu_database(self):
        """
        Downloads the database used by https://arxiv.org/pdf/1904.02726.pdf.
        """
        print("Downloading Liang Yu's disposition CSV...")
        liang_yu_csv_url = 'https://raw.githubusercontent.com/yuliang419/Astronet-Triage/master/astronet/tces.csv'
        response = requests.get(liang_yu_csv_url)
        liang_yu_disposition_path = self.data_directory.joinpath('liang_yu_disposition.csv')
        with open(liang_yu_disposition_path, 'wb') as csv_file:
            csv_file.write(response.content)
        print('Downloading TESS observation list...')
        tess_observations = self.get_all_tess_time_series_observations()
        single_sector_observations = self.get_single_sector_observations(tess_observations)
        single_sector_observations = self.add_sector_column_based_on_single_sector_obs_id(single_sector_observations)
        single_sector_observations['tic_id'] = single_sector_observations['target_name'].astype(int)
        print("Downloading lightcurves which appear in Liang Yu's disposition...")
        # noinspection SpellCheckingInspection
        columns_to_use = ['tic_id', 'Disposition', 'Epoc', 'Period', 'Duration', 'Sectors']
        liang_yu_disposition = pd.read_csv(liang_yu_disposition_path, usecols=columns_to_use)
        liang_yu_observations = pd.merge(single_sector_observations, liang_yu_disposition, how='inner',
                                         left_on=['tic_id', 'sector'], right_on=['tic_id', 'Sectors'])
        number_of_observations_not_found = liang_yu_disposition.shape[0] - liang_yu_observations.shape[0]
        print(f"{liang_yu_observations.shape[0]} observations found that match Liang Yu's entries.")
        print(f'Liang Yu used the FFIs, not the lightcurve products, so many will be missing.')
        print(f"No observations found for {number_of_observations_not_found} entries in Liang Yu's disposition.")
        liang_yu_data_products = self.get_data_products(liang_yu_observations)
        liang_yu_lightcurve_data_products = liang_yu_data_products[
            liang_yu_data_products['productFilename'].str.endswith('lc.fits')
        ]
        download_manifest = self.download_products(liang_yu_lightcurve_data_products)
        print('Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')


if __name__ == '__main__':
    LiangYuLightcurveDatabase().download_liang_yu_database()
