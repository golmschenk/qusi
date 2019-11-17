"""
Code to represent the database of TESS transit data based on disposition tables.
This database is based on the disposition source data is given by `Liang Yu's work
<https://arxiv.org/pdf/1904.02726.pdf>`_.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from pathlib import Path

from ramjet.photometric_database.tess_transit_lightcurve_label_per_time_step_database import \
    TessTransitLightcurveLabelPerTimeStepDatabase


class YuLightcurveDatabase(TessTransitLightcurveLabelPerTimeStepDatabase):
    """
    A class to represent the database of TESS transit data based on disposition tables.
    """

    def __init__(self, data_directory='data/tess_yu'):
        super().__init__(data_directory=data_directory)
        self.liang_yu_dispositions_path = self.data_directory.joinpath('liang_yu_dispositions.csv')

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets.

        :return: The training and validation datasets.
        """
        self.obtain_meta_data_frame_for_available_lightcurves()
        positive_example_paths = self.meta_data_frame[self.meta_data_frame['disposition'] == 'PC']['lightcurve_path']
        print(f'{len(positive_example_paths)} positive examples.')
        negative_example_paths = self.meta_data_frame[self.meta_data_frame['disposition'] != 'PC']['lightcurve_path']
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
        training_preprocessor = lambda file_path: tuple(tf.py_function(self.training_preprocessing,
                                                                       [file_path], [tf.float32, tf.float32]))
        training_dataset = training_dataset.map(training_preprocessor, num_parallel_calls=16)
        training_dataset = training_dataset.padded_batch(self.batch_size, padded_shapes=([None, 2], [None])).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        validation_preprocessor = lambda file_path: tuple(tf.py_function(self.evaluation_preprocessing,
                                                                         [file_path], [tf.float32, tf.float32]))
        validation_dataset = validation_dataset.map(validation_preprocessor, num_parallel_calls=4)
        validation_dataset = validation_dataset.padded_batch(1, padded_shapes=([None, 2], [None])).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset

    def is_positive(self, example_path):
        """
        Checks if an example contains a transit event or not.

        :param example_path: The path to the example to check.
        :return: Whether or not the example contains a transit event.
        """
        candidate_planets = self.meta_data_frame[self.meta_data_frame['disposition'] == 'PC']
        return example_path in candidate_planets['lightcurve_path'].values

    def obtain_meta_data_frame_for_available_lightcurves(self):
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
        liang_yu_dispositions = liang_yu_dispositions[(liang_yu_dispositions['disposition'] != 'PC') |
                                                      (liang_yu_dispositions['transit_epoch'].notna() &
                                                       liang_yu_dispositions['transit_period'].notna() &
                                                       liang_yu_dispositions['transit_duration'].notna())]
        lightcurve_paths = list(self.lightcurve_directory.glob('*lc.fits'))
        tic_ids = [int(self.get_tic_id_from_single_sector_obs_id(path.name)) for path in lightcurve_paths]
        sectors = [self.get_sector_from_single_sector_obs_id(path.name) for path in lightcurve_paths]
        lightcurve_meta_data = pd.DataFrame({'lightcurve_path': list(map(str, lightcurve_paths)), 'tic_id': tic_ids,
                                             'sector': sectors})
        meta_data_frame_with_candidate_nans = pd.merge(liang_yu_dispositions, lightcurve_meta_data,
                                                       how='inner', on=['tic_id', 'sector'])
        self.meta_data_frame = meta_data_frame_with_candidate_nans.dropna()

    def download_liang_yu_database(self):
        """
        Downloads the database used in `Liang Yu's work <https://arxiv.org/pdf/1904.02726.pdf>`_.
        """
        print('Clearing data directory...')
        self.clear_data_directory()
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
        liang_yu_data_products = self.get_product_list(liang_yu_observations)
        liang_yu_lightcurve_data_products = liang_yu_data_products[
            liang_yu_data_products['productFilename'].str.endswith('lc.fits')
        ]
        download_manifest = self.download_products(liang_yu_lightcurve_data_products)
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')

    def generate_label(self, example_path: str, times: np.float32) -> np.bool:
        """
        Generates a label for each time step defining whether or not a transit is occurring.

        :param example_path: The path of the lightcurve file (to determine which row of the meta data to use).
        :param times: The times of the measurements in the lightcurve.
        :return: A boolean label for each time step specifying if transiting is occurring at that time step.
        """
        with np.errstate(all='raise'):
            try:
                example_meta_data = self.meta_data_frame[self.meta_data_frame['lightcurve_path'] == example_path].iloc[
                    0]
                epoch_times = times - example_meta_data['transit_epoch']
                transit_duration = example_meta_data['transit_duration'] / 24  # Convert from hours to days.
                transit_period = example_meta_data['transit_period']
                is_transiting = ((epoch_times + (transit_duration / 2)) % transit_period) < transit_duration
            except FloatingPointError as error:
                print(example_path)
                raise error
        return is_transiting


if __name__ == '__main__':
    YuLightcurveDatabase().download_liang_yu_database()
