"""Code for representing a dataset of lightcurves for binary classification with a single label per time step."""
import numpy as np
import pandas as pd
from pathlib import Path

from photometric_database.lightcurve_database import LightcurveDatabase


class MicrolensingLabelPerTimeStepDatabase(LightcurveDatabase):
    """A representing a dataset of lightcurves for binary classification with a single label per time step."""

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

    @staticmethod
    def get_meta_data_for_lightcurve_file_path(lightcurve_file_path: str,
                                               meta_data_frame: pd.DataFrame) -> pd.Series:
        """
        Gets the meta data for a lightcurve based on the file name from the meta data frame.

        :param lightcurve_file_path: The lightcurve file path.
        :param meta_data_frame: The meta data frame.
        :return: The lightcurve meta data.
        """
        lightcurve_file_name_stem = Path(lightcurve_file_path).stem.split('.')[0]  # Remove all extensions
        field, _, chip, sub_frame, id_ = lightcurve_file_name_stem.split('-')
        # noinspection SpellCheckingInspection
        lightcurve_meta_data = meta_data_frame[(meta_data_frame['ID'] == int(id_)) &
                                               (meta_data_frame['field'] == field) &
                                               (meta_data_frame['chip'] == int(chip)) &
                                               (meta_data_frame['nsub'] == int(sub_frame))].iloc[0]
        return lightcurve_meta_data

    def calculate_magnitudes_for_lightcurve(self, lightcurve_file_path: str,
                                            meta_data_frame: pd.DataFrame) -> np.float32:
        lightcurve_data_frame = pd.read_feather(lightcurve_file_path)
        observation_times = lightcurve_data_frame.HJD.values
        lightcurve_meta_data = self.get_meta_data_for_lightcurve_file_path(lightcurve_file_path, meta_data_frame)
        magnitudes = self.calculate_magnification(observation_time=observation_times,
                                                  minimum_separation_time=lightcurve_meta_data['t0'],
                                                  minimum_einstein_separation=lightcurve_meta_data['umin'],
                                                  einstein_crossing_time=lightcurve_meta_data['tE'])
        return magnitudes
