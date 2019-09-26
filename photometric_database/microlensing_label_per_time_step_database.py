"""Code for representing a dataset of lightcurves for binary classification with a single label per time step."""
from typing import Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from photometric_database.lightcurve_database import LightcurveDatabase


class MicrolensingLabelPerTimeStepDatabase(LightcurveDatabase):
    """A representing a dataset of lightcurves for binary classification with a single label per time step."""

    def __init__(self):
        super().__init__()
        self.meta_data_frame: Union[pd.DataFrame, None] = None
        self.time_steps_per_example = 2000
        self.time_steps_per_validation_example = 30000

    def generate_datasets(self, positive_data_directory: str, negative_data_directory: str,
                          meta_data_file_path: str) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets.

        :param positive_data_directory: The path to the directory containing the positive example files.
        :param negative_data_directory: The path to the directory containing the negative example files.
        :param meta_data_file_path: The path to the microlensing meta data file.
        :return: The training and validation datasets.
        """
        self.meta_data_frame = pd.read_feather(meta_data_file_path)
        positive_example_paths = list(Path(positive_data_directory).glob('*.feather'))
        positive_example_paths = self.remove_file_paths_with_no_meta_data(positive_example_paths, self.meta_data_frame)
        print(f'{len(positive_example_paths)} positive examples.')
        negative_example_paths = list(Path(negative_data_directory).glob('*.feather'))
        print(f'{len(negative_example_paths)} negative examples.')
        positive_datasets = self.get_training_and_validation_datasets_for_file_paths(positive_example_paths)
        positive_training_dataset, positive_validation_dataset = positive_datasets
        negative_datasets = self.get_training_and_validation_datasets_for_file_paths(negative_example_paths)
        negative_training_dataset, negative_validation_dataset = negative_datasets
        training_dataset = self.get_ratio_enforced_dataset(positive_training_dataset, negative_training_dataset,
                                                           positive_to_negative_data_ratio=1)
        validation_dataset = positive_validation_dataset.concatenate(negative_validation_dataset)
        training_dataset = training_dataset.shuffle(buffer_size=len(list(training_dataset)))
        training_preprocessor = lambda file_path: tuple(tf.py_function(self.training_preprocessing,
                                                                       [file_path], [tf.float32, tf.float32]))
        training_dataset = training_dataset.map(training_preprocessor, num_parallel_calls=16)
        training_dataset = training_dataset.map(self.set_training_shape, num_parallel_calls=16)
        training_dataset = training_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validation_preprocessor = lambda file_path: tuple(tf.py_function(self.validation_preprocessing,
                                                                         [file_path], [tf.float32, tf.float32]))
        validation_dataset = validation_dataset.map(validation_preprocessor, num_parallel_calls=16)
        validation_dataset = validation_dataset.map(self.set_validation_shape, num_parallel_calls=16)
        validation_dataset = validation_dataset.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset

    def training_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data for training.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example, label = self.general_preprocessing(example_path_tensor)
        example, label = example.numpy(), label.numpy()
        example, label = self.make_uniform_length_requiring_positive(example, label, self.time_steps_per_example)
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    def validation_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        example, label = self.general_preprocessing(example_path_tensor)
        example, label = example.numpy(), label.numpy()
        label = np.expand_dims(label, axis=-1)
        example_and_label = np.concatenate([example, label], axis=1)
        example_and_label = self.make_uniform_length(example_and_label, self.time_steps_per_validation_example)
        example, label = example_and_label[:, :2], example_and_label[:, 2]
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    def general_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data for evaluation.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example_path = example_path_tensor.numpy().decode('utf-8')
        example_data_frame = pd.read_feather(example_path, columns=['HJD', 'flux'])
        fluxes = example_data_frame['flux'].values
        fluxes = self.normalize(fluxes)
        times = example_data_frame['HJD'].values
        time_differences = np.diff(times, prepend=times[0])
        example = np.stack([fluxes, time_differences], axis=-1)
        if self.is_positive(example_path):
            lightcurve_microlensing_meta_data = self.get_meta_data_for_lightcurve_file_path(example_path,
                                                                                            self.meta_data_frame)
            label = self.magnification_threshold_label_for_lightcurve_meta_data(times,
                                                                                lightcurve_microlensing_meta_data,
                                                                                threshold=1.1)
        else:
            label = np.zeros_like(fluxes)
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)
    
    def set_training_shape(self, lightcurve: tf.Tensor, label: tf.Tensor):
        """
        Explicitly sets the shapes of the lightcurve and label tensor, otherwise TensorFlow can't infer it.

        :param lightcurve: The lightcurve tensor.
        :param label: The label tensor.
        :return: The lightcurve and label tensor with TensorFlow inferable shapes.
        """
        lightcurve.set_shape([self.time_steps_per_example, 2])
        label.set_shape([self.time_steps_per_example])
        return lightcurve, label
    
    def set_validation_shape(self, lightcurve: tf.Tensor, label: tf.Tensor):
        """
        Explicitly sets the shapes of the lightcurve and label tensor, otherwise TensorFlow can't infer it.

        :param lightcurve: The lightcurve tensor.
        :param label: The label tensor.
        :return: The lightcurve and label tensor with TensorFlow inferable shapes.
        """
        lightcurve.set_shape([self.time_steps_per_validation_example, 2])
        label.set_shape([self.time_steps_per_validation_example])
        return lightcurve, label

    @staticmethod
    def load_microlensing_meta_data(meta_data_file_path: str) -> pd.DataFrame:
        """
        Loads a microlensing meta data file into a Pandas data frame.

        :param meta_data_file_path: The path to the original meta data CSV.
        :return: The meta data frame.
        """
        # noinspection SpellCheckingInspection
        column_names = ['field', 'chip', 'nsub', 'ID', 'RA', 'Dec', 'x', 'y', 'ndata', 'ndetect', 'sigma', 'sumsigma',
                        'redchi2_out', 'sepmin', 'ID_dophot', 'type', 'mag', 'mage', 't0', 'tE', 'umin', 'fs', 'fb',
                        't0e', 'tEe', 'tEe1', 'tEe2', 'umine', 'umine1', 'umine2', 'fse', 'fbe', 'chi2', 't0FS', 'tEFS',
                        'uminFS', 'rhoFS', 'fsFS', 'fbFS', 't0eFS', 'tEeFS', 'tEe1FS', 'tEe2FS', 'umineFS', 'umine1FS',
                        'umine2FS', 'rhoeFS', 'rhoe1FS', 'rhoe2FS', 'fseFS', 'fbeFS', 'chi2FS']
        meta_data_frame = pd.read_csv(meta_data_file_path, comment='#', header=None, delim_whitespace=True,
                                      names=column_names)
        return meta_data_frame

    @staticmethod
    def einstein_normalized_separation_in_direction_of_motion(observation_time: np.float32,
                                                              minimum_separation_time: np.float32,
                                                              einstein_crossing_time: np.float32) -> np.float32:
        r"""
        Gets the einstein normalized separation of the source relative to the minimum separation position due to motion.
        This will be the separation perpendicular to the line between the minimum separation position and the lens.
        Broadcasts for arrays times.
        :math:`u_v = 2 \dfrac{t-t_0}{t_E}`

        :param observation_time: :math:`t`, current time of the observation.
        :param minimum_separation_time: :math:`t_0`, the time the minimum separation between source and lens occurs at.
        :param einstein_crossing_time: :math:`t_E`, the time it would take the source to cross the center of the
                                       Einstein ring
        :return: :math:`u_v`, the separation in the direction of source motion.
        """
        return 2 * (observation_time - minimum_separation_time) / einstein_crossing_time

    def calculate_magnification(self, observation_time: np.float32, minimum_separation_time: np.float32,
                                minimum_einstein_separation: np.float32, einstein_crossing_time: np.float32
                                ) -> np.float32:
        r"""
        Calculates the magnification of a microlensing event for a given time step. Broadcasts for arrays of times.
        Allows an infinite magnification in cases where the separation is zero.
        With :math:`u` as the einstein normalized separation, does
        .. math::
           u_v = 2 \dfrac{t-t_0}{t_E}
           u = \sqrt{u_0^2 + u_v^2}
           A = \dfrac{u^2 + 2}{u \sqrt{u^2 + 4}}

        :param observation_time: :math:`t`, current time of the observation.
        :param minimum_separation_time: :math:`t_0`, the time the minimum separation between source and lens occurs at.
        :param minimum_einstein_separation: :math:`u_0`, the minimum einstein normalized separation.
        :param einstein_crossing_time: :math:`t_E`, the time it would take the source to cross the center of the
                                       Einstein ring
        :return: :math:`A`, the magnification for the passed time step(s).
        """
        separation_in_direction_of_motion = self.einstein_normalized_separation_in_direction_of_motion(
            observation_time=observation_time, minimum_separation_time=minimum_separation_time,
            einstein_crossing_time=einstein_crossing_time
        )
        u = np.linalg.norm([minimum_einstein_separation, separation_in_direction_of_motion], axis=0)
        with np.errstate(divide='ignore'):  # Divide by zero resulting in infinity is ok here.
            magnification = (u ** 2 + 2) / (u * (u ** 2 + 4) ** 0.5)
        return magnification

    def get_meta_data_for_lightcurve_file_path(self, lightcurve_file_path: Union[str, Path],
                                               meta_data_frame: pd.DataFrame) -> pd.Series:
        """
        Gets the meta data for a lightcurve based on the file name from the meta data frame.

        :param lightcurve_file_path: The lightcurve file path.
        :param meta_data_frame: The meta data frame.
        :return: The lightcurve meta data.
        """
        lightcurve_meta_data = self.get_meta_data_frame_for_lightcurve_file_path(lightcurve_file_path, meta_data_frame)
        return lightcurve_meta_data.iloc[0]

    def check_if_meta_data_exists_for_lightcurve_file_path(self, lightcurve_file_path: Union[str, Path],
                                                           meta_data_frame: pd.DataFrame) -> bool:
        """
        Gets the meta data for a lightcurve based on the file name from the meta data frame.

        :param lightcurve_file_path: The lightcurve file path.
        :param meta_data_frame: The meta data frame.
        :return: The lightcurve meta data.
        """
        lightcurve_meta_data = self.get_meta_data_frame_for_lightcurve_file_path(lightcurve_file_path, meta_data_frame)
        return lightcurve_meta_data.shape[0] > 0

    @staticmethod
    def get_meta_data_frame_for_lightcurve_file_path(lightcurve_file_path: Union[str, Path],
                                                     meta_data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the meta data frame containing all rows for a lightcurve based on the file name from the meta data frame.

        :param lightcurve_file_path: The lightcurve file path.
        :param meta_data_frame: The meta data frame.
        :return: The lightcurve meta data frame.
        """
        lightcurve_file_name_stem = Path(lightcurve_file_path).name.split('.')[0]  # Remove all extensions
        field, _, chip, sub_frame, id_ = lightcurve_file_name_stem.split('-')
        # noinspection SpellCheckingInspection
        lightcurve_meta_data = meta_data_frame[(meta_data_frame['ID'] == int(id_)) &
                                               (meta_data_frame['field'] == field) &
                                               (meta_data_frame['chip'] == int(chip)) &
                                               (meta_data_frame['nsub'] == int(sub_frame))]
        return lightcurve_meta_data

    def magnification_threshold_label_for_lightcurve_meta_data(self, observation_times: np.float32,
                                                               lightcurve_meta_data: pd.Series,
                                                               threshold: float) -> np.bool:
        """
        Gets the binary per time step label for a lightcurve based on a microlensing magnification threshold.

        :param observation_times: The observation times to calculate magnifications of.
        :param lightcurve_meta_data: The microlensing meta data for magnifications to be based off.
        :param threshold: The magnification threshold required for a time step to be labeled positive.
        :return: The label containing a binary value per time step.
        """
        magnifications = self.calculate_magnifications_for_lightcurve_meta_data(
            times=observation_times,
            lightcurve_microlensing_meta_data=lightcurve_meta_data
        )
        label = magnifications > threshold
        return label

    def make_uniform_length_requiring_positive(self, example: np.ndarray, label: np.ndarray, length: int
                                               ) -> (np.ndarray, np.ndarray):
        """
        Extracts a random segment from an example which is the length specified by the database.

        :param example: The example to extract a segment from.
        :return: The extracted segment.
        """
        while True:
            example_and_label = np.concatenate([example, np.expand_dims(label, axis=-1)], axis=1)
            example_and_label = self.make_uniform_length(example_and_label, length)
            extracted_example, extracted_label = example_and_label[:, :2], example_and_label[:, 2]
            if not label.any() or extracted_label.any():
                return extracted_example, extracted_label

    def calculate_magnifications_for_lightcurve_meta_data(self, times: np.float32,
                                                          lightcurve_microlensing_meta_data: pd.Series) -> np.float32:
        """
        Calculates the magnification values for a given set of times based on the meta data of a specific lightcurve.

        :param times: The observation times to calculate magnifications for.
        :param lightcurve_microlensing_meta_data: The microlensing meta data for magnifications to be based off.
        :return: An array of the magnification of each time.
        """
        magnifications = self.calculate_magnification(
            observation_time=times,
            minimum_separation_time=lightcurve_microlensing_meta_data['t0'],
            minimum_einstein_separation=lightcurve_microlensing_meta_data['umin'],
            einstein_crossing_time=lightcurve_microlensing_meta_data['tE']
        )
        return magnifications

    def remove_file_paths_with_no_meta_data(self, file_paths: List[Path], meta_data_frame: pd.DataFrame) -> List[Path]:
        """
        Filters file paths on whether or not the meta data frame contains an entry for them.

        :param file_paths: The file paths to be filtered.
        :param meta_data_frame: The meta data frame.
        :return: The filtered file paths containing only paths which appear in the meta data frame.
        """
        filtered_file_paths = []
        for file_path in file_paths:
            if self.check_if_meta_data_exists_for_lightcurve_file_path(file_path, meta_data_frame):
                filtered_file_paths.append(file_path)
        return filtered_file_paths
