"""
Code to represent the database of TESS transit data based on disposition tables.
The primary source for dispositions is from the `ExoFOP TOI table
<https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv>`_.
"""
import pandas as pd
import tensorflow as tf
import requests
from pathlib import Path

from ramjet.photometric_database.py_mapper import map_py_function_to_dataset
from ramjet.photometric_database.tess_data_interface import TessDataInterface
from ramjet.photometric_database.tess_transit_lightcurve_label_per_time_step_database import \
    TessTransitLightcurveLabelPerTimeStepDatabase


class ToiLightcurveDatabase(TessTransitLightcurveLabelPerTimeStepDatabase):
    """
    A class to represent the database of TESS transit data based on disposition tables.
    """

    def __init__(self, data_directory='data/tess_toi'):
        super().__init__(data_directory=data_directory)
        self.liang_yu_dispositions_path = self.data_directory.joinpath('liang_yu_dispositions.csv')
        self.toi_dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets.

        :return: The training and validation datasets.
        """
        self.obtain_meta_data_frame_for_available_lightcurves()
        positive_example_paths = self.meta_data_frame['lightcurve_path'].tolist()
        positive_example_paths = list(set(positive_example_paths))  # Remove duplicates from multi-planet targets.
        print(f'{len(positive_example_paths)} positive examples.')
        all_lightcurve_paths = list(map(str, self.lightcurve_directory.glob('*lc.fits')))
        negative_example_paths = list(set(all_lightcurve_paths) - set(self.meta_data_frame['lightcurve_path'].tolist()))
        print(f'{len(negative_example_paths)} negative examples.')
        positive_datasets = self.get_training_and_validation_datasets_for_file_paths(positive_example_paths)
        positive_training_dataset, positive_validation_dataset = positive_datasets
        negative_datasets = self.get_training_and_validation_datasets_for_file_paths(negative_example_paths)
        negative_training_dataset, negative_validation_dataset = negative_datasets
        training_dataset = self.get_ratio_enforced_dataset(positive_training_dataset, negative_training_dataset,
                                                           positive_to_negative_data_ratio=1)
        validation_dataset = positive_validation_dataset.concatenate(negative_validation_dataset)
        if self.trial_directory is not None:
            self.log_dataset_file_names(training_dataset, dataset_name='training')
            self.log_dataset_file_names(validation_dataset, dataset_name='validation')
        training_dataset = training_dataset.shuffle(buffer_size=len(list(training_dataset)))
        training_dataset = map_py_function_to_dataset(training_dataset, self.training_preprocessing,
                                                      number_of_parallel_calls=16,
                                                      output_types=[tf.float32, tf.float32])
        training_dataset = training_dataset.padded_batch(self.batch_size, padded_shapes=([None, 2], [None])).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        validation_dataset = map_py_function_to_dataset(validation_dataset, self.evaluation_preprocessing,
                                                        number_of_parallel_calls=4,
                                                        output_types=[tf.float32, tf.float32])
        validation_dataset = validation_dataset.padded_batch(1, padded_shapes=([None, 2], [None])).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset

    def obtain_meta_data_frame_for_available_lightcurves(self):
        """
        Prepares the meta data frame with the transit information based on known planet transits.
        """
        dispositions = self.load_toi_dispositions_in_project_format()
        confirmed_planet_dispositions = dispositions[dispositions['disposition'].isin(['CP', 'KP']) &
                                                     dispositions['transit_epoch'].notna() &
                                                     dispositions['transit_period'].notna() &
                                                     dispositions['transit_duration'].notna()]
        lightcurve_paths = list(self.lightcurve_directory.glob('*lc.fits'))
        tess_data_interface = TessDataInterface()
        tic_ids = [tess_data_interface.get_tic_id_from_single_sector_obs_id(path.name) for path in lightcurve_paths]
        sectors = [tess_data_interface.get_sector_from_single_sector_obs_id(path.name) for path in lightcurve_paths]
        lightcurve_meta_data = pd.DataFrame({'lightcurve_path': list(map(str, lightcurve_paths)), 'tic_id': tic_ids,
                                             'sector': sectors})
        meta_data_frame_with_candidate_nans = pd.merge(confirmed_planet_dispositions, lightcurve_meta_data,
                                                       how='inner', on=['tic_id', 'sector'])
        self.meta_data_frame = meta_data_frame_with_candidate_nans.dropna()

    def download_exofop_toi_database(self, number_of_negative_lightcurves_to_download=10000):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_.
        """
        print('Clearing data directory...')
        self.clear_data_directory()
        print("Downloading ExoFOP TOI disposition CSV...")
        toi_csv_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        response = requests.get(toi_csv_url)
        with open(self.toi_dispositions_path, 'wb') as csv_file:
            csv_file.write(response.content)
        print('Downloading TESS observation list...')
        tess_data_interface = TessDataInterface()
        tess_observations = tess_data_interface.get_all_tess_time_series_observations()
        single_sector_observations = tess_data_interface.filter_for_single_sector_observations(tess_observations)
        single_sector_observations = self.add_sector_column_based_on_single_sector_obs_id(single_sector_observations)
        single_sector_observations['tic_id'] = single_sector_observations['target_name'].astype(int)
        print("Downloading lightcurves which are confirmed planets in TOI dispositions...")
        # noinspection SpellCheckingInspection
        toi_dispositions = self.load_toi_dispositions_in_project_format()
        confirmed_planet_dispositions = toi_dispositions[toi_dispositions['disposition'].isin(['CP', 'KP'])]
        confirmed_planet_observations = pd.merge(single_sector_observations, confirmed_planet_dispositions, how='inner',
                                                 on=['tic_id', 'sector'])
        observations_not_found = confirmed_planet_dispositions.shape[0] - confirmed_planet_observations.shape[0]
        print(f"{confirmed_planet_observations.shape[0]} observations found that match the TOI dispositions.")
        print(f"No observations found for {observations_not_found} entries in TOI dispositions.")
        confirmed_planet_data_products = tess_data_interface.get_product_list(confirmed_planet_observations)
        confirmed_planet_lightcurve_data_products = confirmed_planet_data_products[
            confirmed_planet_data_products['productFilename'].str.endswith('lc.fits')
        ]
        confirmed_planet_download_manifest = tess_data_interface.download_products(
            confirmed_planet_lightcurve_data_products, data_directory=self.data_directory)
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in confirmed_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print("Downloading lightcurves which are not in TOI dispositions and do not have TCEs (not planets)...")
        print(f'Download limited to {number_of_negative_lightcurves_to_download} lightcurves...')
        # noinspection SpellCheckingInspection
        toi_tic_ids = toi_dispositions['tic_id'].values
        not_toi_observations = single_sector_observations[
            ~single_sector_observations['tic_id'].isin(toi_tic_ids)  # Don't include even false positives.
        ]
        not_toi_observations = not_toi_observations.sample(frac=1, random_state=0)
        # Shorten product list obtaining.
        not_toi_observations = not_toi_observations.head(number_of_negative_lightcurves_to_download * 2)
        not_toi_data_products = tess_data_interface.get_product_list(not_toi_observations)
        not_toi_data_products = self.add_tic_id_column_based_on_single_sector_obs_id(not_toi_data_products)
        not_toi_lightcurve_data_products = not_toi_data_products[
            not_toi_data_products['productFilename'].str.endswith('lc.fits')
        ]
        not_toi_data_validation_data_products = not_toi_data_products[
            not_toi_data_products['productFilename'].str.endswith('dvr.xml')
        ]
        tic_ids_with_dv = not_toi_data_validation_data_products['tic_id'].values
        not_planet_lightcurve_data_products = not_toi_lightcurve_data_products[
            ~not_toi_lightcurve_data_products['tic_id'].isin(tic_ids_with_dv)  # Remove any lightcurves with TCEs.
        ]
        # Shuffle rows.
        not_planet_lightcurve_data_products = not_planet_lightcurve_data_products.sample(frac=1, random_state=0)
        not_planet_download_manifest = tess_data_interface.download_products(
            not_planet_lightcurve_data_products.head(number_of_negative_lightcurves_to_download),
            data_directory=self.data_directory
        )
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in not_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')

    def load_toi_dispositions_in_project_format(self) -> pd.DataFrame:
        """
        Loads the ExoFOP TOI table information from CSV to a data frame using a project consistent naming scheme.

        :return:
        """
        columns_to_use = ['TIC ID', 'TFOPWG Disposition', 'Planet Num', 'Epoch (BJD)', 'Period (days)',
                          'Duration (hours)', 'Sectors']
        dispositions = pd.read_csv(self.toi_dispositions_path, usecols=columns_to_use)
        dispositions.rename(columns={'TIC ID': 'tic_id', 'TFOPWG Disposition': 'disposition',
                                     'Planet Num': 'planet_number', 'Epoch (BJD)': 'transit_epoch',
                                     'Period (days)': 'transit_period', 'Duration (hours)': 'transit_duration',
                                     'Sectors': 'sector'}, inplace=True)
        dispositions['sector'] = dispositions['sector'].str.split(',')
        dispositions = dispositions.explode('sector')
        dispositions['sector'] = pd.to_numeric(dispositions['sector']).astype(pd.Int64Dtype())
        return dispositions


if __name__ == '__main__':
    ToiLightcurveDatabase().download_exofop_toi_database()
