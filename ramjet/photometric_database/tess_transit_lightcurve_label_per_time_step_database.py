"""
Code for a database of TESS transit lightcurves with a label per time step.
"""
import shutil
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits

from ramjet.photometric_database.lightcurve_label_per_time_step_database import LightcurveLabelPerTimeStepDatabase
from ramjet.photometric_database.tess_data_interface import TessDataInterface


class TessTransitLightcurveLabelPerTimeStepDatabase(LightcurveLabelPerTimeStepDatabase):
    """
    A class for a database of TESS transit lightcurves with a label per time step.
    """

    def __init__(self, data_directory='data/tess'):
        super().__init__(data_directory=data_directory)
        self.meta_data_frame: Union[pd.DataFrame, None] = None
        self.lightcurve_directory = self.data_directory.joinpath('lightcurves')
        self.data_validation_directory = self.data_directory.joinpath('data_validations')
        self.data_validation_dictionary = None

    def create_data_directories(self):
        """
        Creates the data directories to be used by the database.
        """
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.lightcurve_directory.mkdir(parents=True, exist_ok=True)
        self.data_validation_directory.mkdir(parents=True, exist_ok=True)

    def clear_data_directory(self):
        """
        Empties the data directory.
        """
        if self.data_directory.exists():
            shutil.rmtree(self.data_directory)
        self.create_data_directories()

    def get_lightcurve_file_paths(self) -> List[Path]:
        """
        Gets all the file paths for the available lightcurves.
        """
        return list(self.lightcurve_directory.glob('*.fits'))

    def is_positive(self, example_path):
        """
        Checks if an example contains a transit event or not.

        :param example_path: The path to the example to check.
        :return: Whether or not the example contains a transit event.
        """
        return example_path in self.meta_data_frame['lightcurve_path'].values

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

    def general_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example_path = example_path_tensor.numpy().decode('utf-8')
        tess_data_interface = TessDataInterface()
        fluxes, times = tess_data_interface.load_fluxes_and_times_from_fits_file(example_path)
        fluxes = self.normalize(fluxes)
        time_differences = np.diff(times, prepend=times[0])
        example = np.stack([fluxes, time_differences], axis=-1)
        if self.is_positive(example_path):
            label = self.generate_label(example_path, times)
        else:
            label = np.zeros_like(fluxes)
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)
