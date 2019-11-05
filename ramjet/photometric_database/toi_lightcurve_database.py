"""
Code to represent the database of TESS transit data based on disposition tables.
The primary source for dispositions is from the `ExoFOP TOI table
<https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv>`_.
"""
from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from pathlib import Path
from astropy.io import fits

from ramjet.photometric_database.tess_transit_lightcurve_label_per_time_step_database import \
    TessTransitLightcurveLabelPerTimeStepDatabase


class ToiLightcurveDatabase(TessTransitLightcurveLabelPerTimeStepDatabase):
    """
    A class to represent the database of TESS transit data based on disposition tables.
    """

    def __init__(self, data_directory='data/tess'):
        super().__init__(data_directory=data_directory)
        self.liang_yu_dispositions_path = self.data_directory.joinpath('liang_yu_dispositions.csv')
        self.toi_dispositions_path = self.data_directory.joinpath('toi_dispositions.csv')
        self.meta_data_frame: Union[pd.DataFrame, None] = None

    def generate_datasets(self, positive_data_directory: str, negative_data_directory: str,
                          meta_data_file_path: str) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets.

        :param positive_data_directory: The path to the directory containing the positive example files.
        :param negative_data_directory: The path to the directory containing the negative example files.
        :param meta_data_file_path: The path to the microlensing meta data file.
        :return: The training and validation datasets.
        """
        self.obtain_meta_data_frame_for_available_lightcurves()
        positive_example_paths = self.meta_data_frame['lightcurve_path'].tolist()
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

    def general_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example_path = example_path_tensor.numpy().decode('utf-8')
        fluxes, times = self.load_fluxes_and_times_from_fits_file(example_path)
        fluxes = self.normalize(fluxes)
        time_differences = np.diff(times, prepend=times[0])
        example = np.stack([fluxes, time_differences], axis=-1)
        if self.is_positive(example_path):
            label = self.generate_label(example_path, times)
        else:
            label = np.zeros_like(fluxes)
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    @staticmethod
    def load_fluxes_and_times_from_fits_file(example_path: Union[str, Path]) -> (np.ndarray, np.ndarray):
        """
        Extract the flux and time values from a TESS FITS file.

        :param example_path: The path to the FITS file.
        :return: The flux and times values from the FITS file.
        """
        hdu_list = fits.open(example_path)
        lightcurve = hdu_list[1].data  # Lightcurve information is in first extension table.
        fluxes = lightcurve['SAP_FLUX']
        times = lightcurve['TIME']
        assert times.shape == fluxes.shape
        # noinspection PyUnresolvedReferences
        nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.argwhere(np.isnan(times)))
        fluxes = np.delete(fluxes, nan_indexes)
        times = np.delete(times, nan_indexes)
        return fluxes, times

    def generate_label(self, example_path: str, times: np.float32) -> np.bool:
        """
        Generates a label for each time step defining whether or not a transit is occurring.

        :param example_path: The path of the lightcurve file (to determine which row of the meta data to use).
        :param times: The times of the measurements in the lightcurve.
        :return: A boolean label for each time step specifying if transiting is occurring at that time step.
        """
        any_planet_is_transiting = np.zeros_like(times, dtype=np.bool)
        with np.errstate(all='raise'):
            try:
                planets_meta_data = self.meta_data_frame[self.meta_data_frame['lightcurve_path'] == example_path]
                for index, planet_meta_data in planets_meta_data.iterrows():
                    transit_tess_epoch = planet_meta_data['transit_epoch'] - 2457000  # Offset of BJD to BTJD
                    epoch_times = times - transit_tess_epoch
                    transit_duration = planet_meta_data['transit_duration'] / 24  # Convert from hours to days.
                    transit_period = planet_meta_data['transit_period']
                    half_duration = transit_duration / 2
                    if transit_period == 0:  # Single transit known, no repeating signal.
                        planet_is_transiting = (-half_duration < epoch_times) & (epoch_times < half_duration)
                    else:  # Period known, signal should repeat every period.
                        planet_is_transiting = ((epoch_times + half_duration) % transit_period) < transit_duration
                    any_planet_is_transiting = any_planet_is_transiting | planet_is_transiting
            except FloatingPointError as error:
                print(example_path)
                raise error
        return any_planet_is_transiting

    def is_positive(self, example_path):
        """
        Checks if an example contains a transit event or not.

        :param example_path: The path to the example to check.
        :return: Whether or not the example contains a transit event.
        """
        return example_path in self.meta_data_frame['lightcurve_path'].values

    def obtain_meta_data_frame_for_available_lightcurves(self):
        dispositions = self.load_toi_dispositions_in_project_format()
        confirmed_planet_dispositions = dispositions[dispositions['disposition'].isin(['CP', 'KP']) &
                                                     dispositions['transit_epoch'].notna() &
                                                     dispositions['transit_period'].notna() &
                                                     dispositions['transit_duration'].notna()]
        lightcurve_paths = list(self.lightcurve_directory.glob('*lc.fits'))
        tic_ids = [int(self.get_tic_id_from_single_sector_obs_id(path.name)) for path in lightcurve_paths]
        sectors = [self.get_sector_from_single_sector_obs_id(path.name) for path in lightcurve_paths]
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
        tess_observations = self.get_all_tess_time_series_observations()
        single_sector_observations = self.get_single_sector_observations(tess_observations)
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
        confirmed_planet_data_products = self.get_product_list(confirmed_planet_observations)
        confirmed_planet_lightcurve_data_products = confirmed_planet_data_products[
            confirmed_planet_data_products['productFilename'].str.endswith('lc.fits')
        ]
        confirmed_planet_download_manifest = self.download_products(confirmed_planet_lightcurve_data_products)
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
        not_toi_data_products = self.get_product_list(not_toi_observations)
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
        not_planet_download_manifest = self.download_products(
            not_planet_lightcurve_data_products.head(number_of_negative_lightcurves_to_download)
        )
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in not_planet_download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')

    def load_toi_dispositions_in_project_format(self) -> pd.DataFrame:
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
