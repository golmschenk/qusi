"""
Tests for the LightcurveDatabase class.
"""
from typing import Any
from unittest.mock import Mock
import numpy as np
import tensorflow as tf
import pytest


from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class TestLightcurveDatabase:
    @pytest.fixture
    def database(self):
        """Fixture of an instance of the class under test."""
        return LightcurveDatabase()

    @pytest.fixture
    def module(self) -> Any:
        """Fixture of the module under test."""
        import ramjet.photometric_database.lightcurve_database as lightcurve_database_module
        return lightcurve_database_module

    def test_extraction_of_chunk_and_remainder_from_array(self, database, module):
        module.np.random.shuffle = Mock()
        array_to_chunk = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        expected_chunk = np.array([[3, 3], [4, 4]])
        expected_remainder = np.array([[1, 1], [2, 2], [5, 5], [6, 6]])
        chunk, remainder = database.extract_shuffled_chunk_and_remainder(array_to_chunk, chunk_ratio=1 / 3,
                                                                         chunk_to_extract_index=1)
        assert np.array_equal(chunk, expected_chunk)
        assert np.array_equal(remainder, expected_remainder)


    def test_creating_a_padded_window_dataset_for_a_zipped_example_and_label_dataset(self, database):
        # noinspection PyMissingOrEmptyDocstring
        def examples_generator():
            for example in [[1, 1], [2, 2], [3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]:
                yield example

        # noinspection PyMissingOrEmptyDocstring
        def labels_generator():
            for label in [[-1, -1], [-2, -2], [-3, -3], [-4, -4, -4], [-5, -5, -5], [-6, -6, -6]]:
                yield label

        example_dataset = tf.data.Dataset.from_generator(examples_generator, output_types=tf.float32)
        label_dataset = tf.data.Dataset.from_generator(labels_generator, output_types=tf.float32)
        dataset = tf.data.Dataset.zip((example_dataset, label_dataset))
        padded_window_dataset = database.padded_window_dataset_for_zipped_example_and_label_dataset(
            dataset=dataset, batch_size=3, window_shift=2, padded_shapes=([None], [None]))
        padded_window_iterator = iter(padded_window_dataset)
        batch0 = next(padded_window_iterator)
        assert np.array_equal(batch0[0].numpy(), [[1, 1], [2, 2], [3, 3]])
        batch1 = next(padded_window_iterator)
        assert np.array_equal(batch1[0].numpy(), [[3, 3, 0], [4, 4, 4], [5, 5, 5]])
