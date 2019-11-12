"""Code for representing a database of lightcurves for binary classification with a single label per time step."""
from abc import abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class LightcurveLabelPerTimeStepDatabase(LightcurveDatabase):
    """A representation of a database of lightcurves for binary classification with a single label per time step."""

    def __init__(self, data_directory='data'):
        super().__init__(data_directory=data_directory)
        self.meta_data_frame: Union[pd.DataFrame, None] = None
        self.time_steps_per_example = 6400
        self.length_multiple_base = 32
        self.batch_size = 100

    def training_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data for training.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example, label = self.general_preprocessing(example_path_tensor)
        example, label = example.numpy(), label.numpy()
        example, label = self.make_uniform_length_requiring_positive(
            example, label, self.time_steps_per_example, required_length_multiple_base=self.length_multiple_base
        )
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    def evaluation_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data for evaluation.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example, label = self.general_preprocessing(example_path_tensor)
        example, label = example.numpy(), label.numpy()
        example, label = self.make_uniform_length_requiring_positive(
            example, label, required_length_multiple_base=self.length_multiple_base, evaluation=True
        )
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    @abstractmethod
    def general_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        pass

    def make_uniform_length_requiring_positive(self, example: np.ndarray, label: np.ndarray,
                                               length: Union[int, None] = None,
                                               required_length_multiple_base: Union[int, None] = None,
                                               evaluation: bool = False) -> (np.ndarray, np.ndarray):
        """
        Extracts a random segment from an example of the length specified. For examples with a positive label,
        the segment is required to include at least 1 positive time step. Examples shorter than the specified length
        will be repeated to fit the length.

        :param example: The example to extract a segment from.
        :param label: The label whose matching segment should be extracted.
        :param length: The length to make the example.
        :param required_length_multiple_base: An optional base which the length is rounded to.
        :param evaluation: Whether the script is evaluating (in which case we don't want a random position).
        :return: The extracted segment and corresponding label.
        """
        if length is None:
            length = label.shape[0]
        if required_length_multiple_base is not None:
            length = self.round_to_base(length, base=required_length_multiple_base)
        if length == label.shape[0]:
            return example, label
        if not evaluation and label.shape[0] > length and any(label):
            valid_start_indexes = self.valid_start_indexes_for_segment_including_positive(label.astype(np.bool), length)
            start_index = np.random.choice(valid_start_indexes)
            end_index = start_index + length
            example = example[start_index:end_index]
            label = label[start_index:end_index]
            return example, label
        example_and_label = np.concatenate([example, np.expand_dims(label, axis=-1)], axis=1)
        example_and_label = self.make_uniform_length(example_and_label, length, randomize=not evaluation)
        extracted_example, extracted_label = example_and_label[:, :2], example_and_label[:, 2]
        return extracted_example, extracted_label

    @staticmethod
    def valid_start_indexes_for_segment_including_positive(boolean_array: np.ndarray, segment_length: int):
        """
        Gets all indexes of an array where a segment started at that index will include at least one True entry.
        In other words, an

        :param boolean_array: The array indicating which positions are positive.
        :param segment_length: The length of the segments to consider.
        :return: The valid start indexes.
        """
        assert boolean_array.shape[0] >= segment_length
        for _ in range(segment_length - 1):
            boolean_array = boolean_array | np.roll(boolean_array, -1)
        boolean_array = boolean_array[:-(segment_length - 1)]  # Segments extending beyond the array are invalid.
        return np.where(boolean_array)[0]

    @staticmethod
    def round_to_base(number: int, base: int) -> int:
        """
        Rounds a number to a specific base/multiple.

        :param number: The number to round.
        :param base: The base to round to.
        :return: The rounded number.
        """
        return base * round(number / base)

    @staticmethod
    def inference_postprocessing(label: Union[tf.Tensor, np.ndarray], prediction: Union[tf.Tensor, np.ndarray],
                                 length: int) -> (np.ndarray, np.ndarray):
        """
        Prepares the label and prediction for use alongside the original data. In particular, as the network may
        require a specific multiple size, the label and prediction may need to be slightly clipped or padded. Also
        ensures NumPy types for easy use.

        :param label: The ground truth label (preprocessed for use by the network).
        :param prediction: The prediction from the network.
        :param length: The length of the original example before preprocessing.
        :return: The label and prediction prepared for comparison to the original unpreprocessed example.
        """
        if isinstance(label, tf.Tensor):
            label = label.numpy()
        if isinstance(prediction, tf.Tensor):
            prediction = prediction.numpy()
        if label.shape[0] > length:
            label = label[:length]
            prediction = prediction[:length]
        elif label.shape[0] < length:
            elements_to_repeat = length - label.shape[0]
            label = np.pad(label, (0, elements_to_repeat), mode='constant')
            prediction = np.pad(prediction, (0, elements_to_repeat), mode='constant')
        return label, prediction
