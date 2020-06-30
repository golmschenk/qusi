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

    def test_splitting_of_training_and_validation_datasets_for_file_paths_with_list_input(self, database):
        paths = [Path('a'), Path('b'), Path('c'), Path('d'), Path('e'), Path('f')]
        database.validation_ratio = 1/3
        datasets = database.get_training_and_validation_datasets_for_file_paths(paths)
        training_paths_dataset, validation_paths_dataset = datasets
        assert list(training_paths_dataset) == ['b', 'c', 'e', 'f']
        assert list(validation_paths_dataset) == ['a', 'd']

    def test_splitting_of_training_and_validation_datasets_for_file_paths_with_generator_factory_input(self, database):
        def generator_factory():
            return (Path(string) for string in ['a', 'b', 'c', 'd', 'e', 'f'])
        database.validation_ratio = 1 / 3
        datasets = database.get_training_and_validation_datasets_for_file_paths(generator_factory)
        training_paths_dataset, validation_paths_dataset = datasets
        assert list(training_paths_dataset) == ['b', 'c', 'e', 'f']
        assert list(validation_paths_dataset) == ['a', 'd']

    def test_splitting_of_training_and_validation_datasets_for_file_paths_with_list_factory_input(self, database):
        def generator_factory():
            return [Path(string) for string in ['a', 'b', 'c', 'd', 'e', 'f']]
        database.validation_ratio = 1 / 3
        datasets = database.get_training_and_validation_datasets_for_file_paths(generator_factory)
        training_paths_dataset, validation_paths_dataset = datasets
        assert list(training_paths_dataset) == ['b', 'c', 'e', 'f']
        assert list(validation_paths_dataset) == ['a', 'd']

    def test_training_and_validation_datasets_from_generator_can_repeat(self, database):
        def generator_factory():
            return (Path(string) for string in ['a', 'b', 'c'])
        database.validation_ratio = 1 / 3
        datasets = database.get_training_and_validation_datasets_for_file_paths(generator_factory)
        training_paths_dataset, _ = datasets
        assert list(training_paths_dataset) == ['b', 'c']
        assert list(training_paths_dataset) == ['b', 'c']  # Force the dataset to resolve a second time.

    def test_training_and_validation_datasets_from_generator_do_not_mix_values_on_repeat(self, database):
        def generator_factory():
            return (Path(string) for string in ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        database.validation_ratio = 1 / 3
        datasets = database.get_training_and_validation_datasets_for_file_paths(generator_factory)
        training_paths_dataset, validation_paths_dataset = datasets
        training_paths_dataset = training_paths_dataset.repeat(2)
        validation_paths_dataset = validation_paths_dataset.repeat(2)
        assert len(list(training_paths_dataset)) == 8
        assert len(list(validation_paths_dataset)) == 6
        for element in ['a', 'd', 'g']:
            assert element not in training_paths_dataset
        for element in ['b', 'c', 'e', 'f']:
            assert element not in validation_paths_dataset

    def test_paths_dataset_from_list_or_generator_factory_can_use_a_list(self, database):
        paths = [Path('a'), Path('b'), Path('c'), Path('d'), Path('e'), Path('f')]
        paths_dataset = database.paths_dataset_from_list_or_generator_factory(paths)
        assert list(paths_dataset) == ['a', 'b', 'c', 'd', 'e', 'f']

    def test_paths_dataset_from_list_or_generator_factory_can_use_a_generator_factory(self, database):
        def generator_factory():
            return (Path(string) for string in ['a', 'b', 'c'])
        paths_dataset = database.paths_dataset_from_list_or_generator_factory(generator_factory)
        assert list(paths_dataset) == ['a', 'b', 'c']
        assert list(paths_dataset) == ['a', 'b', 'c']  # Check new generator rather than old empty one.

    def test_normalization_does_not_invert_lightcurve_shape_when_there_are_negative_values(self, database):
        unnormalized_positive_lightcurve_fluxes = np.array([10, 20, 30, 25, 15], dtype=np.float32)
        normalized_positive_lightcurve_fluxes = database.normalize(unnormalized_positive_lightcurve_fluxes)
        assert normalized_positive_lightcurve_fluxes.argmax() == 2
        assert normalized_positive_lightcurve_fluxes.argmin() == 0
        unnormalized_negative_lightcurve_fluxes = np.array([-30, -20, -10, -15, -25], dtype=np.float32)
        normalized_negative_lightcurve_fluxes = database.normalize(unnormalized_negative_lightcurve_fluxes)
        assert normalized_negative_lightcurve_fluxes.argmax() == 2
        assert normalized_negative_lightcurve_fluxes.argmin() == 0

    def test_normalization_brings_values_close_to_negative_one_to_one_range(self, database):
        epsilon = 0.6
        unnormalized_positive_lightcurve_fluxes = np.array([10, 20, 30, 20, 10], dtype=np.float32)
        normalized_positive_lightcurve_fluxes = database.normalize(unnormalized_positive_lightcurve_fluxes)
        assert (normalized_positive_lightcurve_fluxes > (-1 - epsilon)).all()
        assert (normalized_positive_lightcurve_fluxes < (1 + epsilon)).all()
        assert np.min(normalized_positive_lightcurve_fluxes) < (-1 + epsilon)
        assert np.max(normalized_positive_lightcurve_fluxes) > (1 - epsilon)
        unnormalized_negative_lightcurve_fluxes = np.array([-30, -20, -10, -20, -30], dtype=np.float32)
        normalized_negative_lightcurve_fluxes = database.normalize(unnormalized_negative_lightcurve_fluxes)
        assert (normalized_negative_lightcurve_fluxes > (-1 - epsilon)).all()
        assert (normalized_negative_lightcurve_fluxes < (1 + epsilon)).all()
        assert np.min(normalized_negative_lightcurve_fluxes) < (-1 + epsilon)
        assert np.max(normalized_negative_lightcurve_fluxes) > (1 - epsilon)
