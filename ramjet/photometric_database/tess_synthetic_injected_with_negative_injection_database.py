import time
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase
from ramjet.py_mapper import map_py_function_to_dataset


class TessSyntheticInjectedWithNegativeInjectionDatabase(TessSyntheticInjectedDatabase):
    """
    A database where both negative and positive signals are injected. Specifically designed for cases where the
    positive injection uses a real lightcurve converted to a synthetic signal. As this can results in other statistics
    suggesting a positive signal (such as extra noise from the real signal being used as an injection signal), injecting
    a negative signal in a similar fashion can reduce the likelihood the network will train on the wrong features.
    """
    def __init__(self, data_directory='data/self_lensing_binaries'):
        super().__init__(data_directory=data_directory)
        self.allow_out_of_bounds_injection = True

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets for the database.

        :return: The training and validation dataset.
        """
        synthetic_signal_paths_dataset = self.paths_dataset_from_list_or_generator_factory(
            self.get_all_synthetic_signal_paths)
        lightcurve_paths_datasets = self.get_training_and_validation_datasets_for_file_paths(
            self.get_all_lightcurve_paths)
        training_lightcurve_paths_dataset, validation_lightcurve_paths_dataset = lightcurve_paths_datasets
        negative_synthetic_signal_paths_dataset, _ = self.get_training_and_validation_datasets_for_file_paths(
            self.get_all_lightcurve_paths)
        shuffled_training_lightcurve_paths_dataset = training_lightcurve_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_synthetic_signal_paths_dataset = synthetic_signal_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        shuffled_negative_synthetic_signal_paths_dataset = negative_synthetic_signal_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        zipped_training_paths_dataset = tf.data.Dataset.zip((shuffled_training_lightcurve_paths_dataset,
                                                             shuffled_synthetic_signal_paths_dataset,
                                                             shuffled_negative_synthetic_signal_paths_dataset))
        output_types = (tf.float32, tf.float32)
        output_shapes = [(self.time_steps_per_example, 1), (1,)]
        lightcurve_training_dataset = map_py_function_to_dataset(zipped_training_paths_dataset,
                                                                 self.positive_and_negative_injection_preprocessing,
                                                                 self.number_of_parallel_processes_per_map,
                                                                 output_types=output_types,
                                                                 output_shapes=output_shapes,
                                                                 flat_map=True)
        batched_training_dataset = self.window_dataset_for_zipped_example_and_label_dataset(lightcurve_training_dataset,
                                                                                            self.batch_size,
                                                                                            self.batch_size // 10)
        prefetch_training_dataset = batched_training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        shuffled_validation_lightcurve_paths_dataset = validation_lightcurve_paths_dataset.repeat().shuffle(
            buffer_size=self.shuffle_buffer_size)
        zipped_validation_paths_dataset = tf.data.Dataset.zip((shuffled_validation_lightcurve_paths_dataset,
                                                               shuffled_synthetic_signal_paths_dataset,
                                                               shuffled_negative_synthetic_signal_paths_dataset))
        lightcurve_validation_dataset = map_py_function_to_dataset(zipped_validation_paths_dataset,
                                                                   self.positive_and_negative_injection_preprocessing,
                                                                   self.number_of_parallel_processes_per_map,
                                                                   output_types=output_types,
                                                                   output_shapes=output_shapes,
                                                                   flat_map=True)
        batched_validation_dataset = lightcurve_validation_dataset.batch(self.batch_size)
        prefetch_validation_dataset = batched_validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return prefetch_training_dataset, prefetch_validation_dataset

    def positive_and_negative_injection_preprocessing(self, lightcurve_path_tensor: tf.Tensor,
                                                      positive_synthetic_signal_path_tensor: tf.Tensor,
                                                      negative_synthetic_signal_path_tensor: tf.Tensor
                                                      ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        The preprocesses examples where a lightcurve is injected with a positive signal and a negative signal to produce
        a positive and negative pair.

        :param lightcurve_path_tensor: The lightcurve's path to be preprocessed.
        :param positive_synthetic_signal_path_tensor: The path of the positive synthetic signal to be injected.
        :param negative_synthetic_signal_path_tensor: The path of the negative synthetic signal to be injected.
        :return: Two examples, one negative, injected with the negative signal and one positive, injected with the
                 positive signal (paired as a tuple); and the corresponding labels (paired as a tuple). Expected to have
                 a post flat mapping to make each element of the data be an individual example and label pair.
        """
        lightcurve_path = lightcurve_path_tensor.numpy().decode('utf-8')
        synthetic_signal_path = positive_synthetic_signal_path_tensor.numpy().decode('utf-8')
        negative_synthetic_signal_path = negative_synthetic_signal_path_tensor.numpy().decode('utf-8')
        fluxes, times = self.load_fluxes_and_times_from_lightcurve_path(lightcurve_path)
        positive_synthetic_signal = self.load_magnifications_and_times_from_synthetic_signal_path(synthetic_signal_path)
        positive_synthetic_magnifications, positive_synthetic_times = positive_synthetic_signal
        negative_synthetic_signal = self.load_magnifications_and_times_from_negative_synthetic_signal_path(
            negative_synthetic_signal_path)
        negative_synthetic_magnifications, negative_synthetic_times = negative_synthetic_signal
        fluxes_with_injected_signal = self.inject_signal_into_lightcurve(
            fluxes, times, positive_synthetic_magnifications, positive_synthetic_times)
        fluxes = self.inject_signal_into_lightcurve(fluxes, times, negative_synthetic_magnifications,
                                                    negative_synthetic_times)
        time_seed = int(time.time())
        fluxes = self.flux_preprocessing(fluxes, seed=time_seed)
        fluxes_with_injected_signal = self.flux_preprocessing(fluxes_with_injected_signal, seed=time_seed)
        lightcurve = np.expand_dims(fluxes, axis=-1)
        lightcurve_with_injected_signal = np.expand_dims(fluxes_with_injected_signal, axis=-1)
        examples = (lightcurve, lightcurve_with_injected_signal)
        labels = (np.array([0]), np.array([1]))
        return examples, labels

    def load_magnifications_and_times_from_negative_synthetic_signal_path(self, negative_synthetic_signal_path: str
                                                                          ) -> (np.ndarray, np.ndarray):
        """
        Loads the negative synthetic signals.

        :param negative_synthetic_signal_path: The path of the negative synthetic signal.
        :return: The magnifications and the times of the negative signal.
        """
        fluxes, times = self.load_fluxes_and_times_from_lightcurve_path(negative_synthetic_signal_path)
        synthetic_magnifications, synthetic_times = self.generate_synthetic_signal_from_real_data(fluxes, times)
        return synthetic_magnifications, synthetic_times

    def get_all_negative_synthetic_signal_paths(self) -> Iterable[Path]:
        """
        Returns the iterable of all negative synthetic signals to use. Expected to be overridden for subclass databases.

        :return: The iterable of negative synthetic signals paths.
        """
        negative_synthetic_signal_paths = self.get_all_lightcurve_paths()
        return negative_synthetic_signal_paths
