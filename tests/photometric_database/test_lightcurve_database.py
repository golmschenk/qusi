"""
Tests for the LightcurveDatabase class.
"""
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
import numpy as np
import tensorflow as tf
import pytest

import ramjet.photometric_database.lightcurve_database
from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class TestLightcurveDatabase:
    @pytest.fixture
    def database(self):
        """Fixture of an instance of the class under test."""
        return LightcurveDatabase()

    @pytest.fixture
    def database_module(self) -> Any:
        import ramjet.photometric_database.lightcurve_database as database_module
        return database_module

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

    @patch.object(ramjet.photometric_database.lightcurve_database.np.random, 'randint')
    def test_lightcurve_padding_can_be_made_non_random_for_evaluation(self, mock_randint, database, database_module):
        mock_randint.return_value = 3
        lightcurve0 = database.make_uniform_length(np.array([10, 20, 30, 40, 50]), length=9, randomize=True)
        assert np.array_equal(lightcurve0, [30, 40, 50, 10, 20, 30, 40, 50, 10])
        lightcurve1 = database.make_uniform_length(np.array([10, 20, 30, 40, 50]), length=9, randomize=False)
        assert np.array_equal(lightcurve1, [10, 20, 30, 40, 50, 10, 20, 30, 40])
        # Should also work for lightcurves with more than just 1 value over time.
        lightcurve2 = database.make_uniform_length(np.array([[10], [20], [30], [40], [50]]), length=9, randomize=True)
        assert np.array_equal(lightcurve2, [[30], [40], [50], [10], [20], [30], [40], [50], [10]])
        lightcurve3 = database.make_uniform_length(np.array([[10], [20], [30], [40], [50]]), length=9, randomize=False)
        assert np.array_equal(lightcurve3, [[10], [20], [30], [40], [50], [10], [20], [30], [40]])

    @patch.object(ramjet.photometric_database.lightcurve_database.np.random, 'shuffle')
    def test_splitting_of_training_and_validation_datasets_for_file_paths(self, mock_shuffle, database):
        mock_shuffle = Mock()  # Make sure the shuffle does nothing to get consistent output.
        paths = [Path('a'), Path('b'), Path('c'), Path('d'), Path('e'), Path('f')]
        database.validation_ratio = 1/3
        datasets = database.get_training_and_validation_datasets_for_file_paths(paths)
        training_paths_dataset, validation_paths_dataset = datasets
        assert list(training_paths_dataset) == ['c', 'd', 'e', 'f']
        assert list(validation_paths_dataset) == ['a', 'b']
