"""
Tests for the LightCurveDatabase class.
"""
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
import numpy as np
import tensorflow as tf
import pytest

import ramjet.photometric_database.light_curve_database as module
from ramjet.photometric_database.light_curve_database import LightCurveDatabase


class TestLightCurveDatabase:
    @pytest.fixture
    def database(self):
        """Fixture of an instance of the class under test."""
        return LightCurveDatabase()

    @pytest.fixture
    def database_module(self) -> Any:
        import ramjet.photometric_database.light_curve_database as database_module
        return database_module

    @pytest.fixture
    def module(self) -> Any:
        """Fixture of the module under test."""
        import ramjet.photometric_database.light_curve_database as light_curve_database_module
        return light_curve_database_module

    def test_extraction_of_chunk_and_remainder_from_array(self, database, module):
        module.np.random.shuffle = Mock()
        array_to_chunk = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        expected_chunk = np.array([[3, 3], [4, 4]])
        expected_remainder = np.array([[1, 1], [2, 2], [5, 5], [6, 6]])
        chunk, remainder = database.extract_shuffled_chunk_and_remainder(array_to_chunk, chunk_ratio=1 / 3,
                                                                         chunk_to_extract_index=1)
        assert np.array_equal(chunk, expected_chunk)
        assert np.array_equal(remainder, expected_remainder)

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

    def test_normalization_does_not_invert_light_curve_shape_when_there_are_negative_values(self, database):
        unnormalized_positive_light_curve_fluxes = np.array([10, 20, 30, 25, 15], dtype=np.float32)
        normalized_positive_light_curve_fluxes = database.normalize_on_percentiles(unnormalized_positive_light_curve_fluxes)
        assert normalized_positive_light_curve_fluxes.argmax() == 2
        assert normalized_positive_light_curve_fluxes.argmin() == 0
        unnormalized_negative_light_curve_fluxes = np.array([-30, -20, -10, -15, -25], dtype=np.float32)
        normalized_negative_light_curve_fluxes = database.normalize_on_percentiles(unnormalized_negative_light_curve_fluxes)
        assert normalized_negative_light_curve_fluxes.argmax() == 2
        assert normalized_negative_light_curve_fluxes.argmin() == 0

    def test_normalization_brings_values_close_to_negative_one_to_one_range(self, database):
        epsilon = 0.6
        unnormalized_positive_light_curve_fluxes = np.array([10, 20, 30, 20, 10], dtype=np.float32)
        normalized_positive_light_curve_fluxes = database.normalize_on_percentiles(unnormalized_positive_light_curve_fluxes)
        assert (normalized_positive_light_curve_fluxes > (-1 - epsilon)).all()
        assert (normalized_positive_light_curve_fluxes < (1 + epsilon)).all()
        assert np.min(normalized_positive_light_curve_fluxes) < (-1 + epsilon)
        assert np.max(normalized_positive_light_curve_fluxes) > (1 - epsilon)
        unnormalized_negative_light_curve_fluxes = np.array([-30, -20, -10, -20, -30], dtype=np.float32)
        normalized_negative_light_curve_fluxes = database.normalize_on_percentiles(unnormalized_negative_light_curve_fluxes)
        assert (normalized_negative_light_curve_fluxes > (-1 - epsilon)).all()
        assert (normalized_negative_light_curve_fluxes < (1 + epsilon)).all()
        assert np.min(normalized_negative_light_curve_fluxes) < (-1 + epsilon)
        assert np.max(normalized_negative_light_curve_fluxes) > (1 - epsilon)

    def test_percentile_normalization_gives_exact_results(self, database):
        unnormalized_light_curve_fluxes = np.linspace(0, 100, num=101, dtype=np.float32)
        normalized_light_curve_fluxes = database.normalize_on_percentiles(unnormalized_light_curve_fluxes)
        assert normalized_light_curve_fluxes[10] == -1
        assert normalized_light_curve_fluxes[90] == 1

    def test_percentile_normalization_zeroes_light_curve_with_all_same_value(self, database):
        unnormalized_light_curve_fluxes = np.full(shape=[100], fill_value=50)
        normalized_light_curve_fluxes = database.normalize_on_percentiles(unnormalized_light_curve_fluxes)
        assert normalized_light_curve_fluxes[10] == 0
        assert normalized_light_curve_fluxes[90] == 0

    def test_percentile_normalization_can_normalize_an_array_and_errors(self, database):
        unnormalized_fluxes = np.linspace(0, 100, num=101, dtype=np.float32)
        unnormalized_flux_errors = np.linspace(0, 10, num=101, dtype=np.float32)
        normalized_fluxes, normalized_flux_errors = database.normalize_on_percentiles_with_errors(
            array=unnormalized_fluxes, array_errors=unnormalized_flux_errors)
        assert normalized_fluxes[10] == -1
        assert normalized_fluxes[90] == 1
        assert normalized_flux_errors[10] == pytest.approx(0.025)
        assert normalized_flux_errors[90] == pytest.approx(0.225)

    def test_percentile_normalization_zeroes_when_array_is_all_same_value_with_errors(self, database):
        unnormalized_fluxes = np.full(shape=[100], fill_value=50)
        unnormalized_flux_errors = np.linspace(0, 10, num=101, dtype=np.float32)
        normalized_fluxes, normalized_flux_errors = database.normalize_on_percentiles_with_errors(
            array=unnormalized_fluxes, array_errors=unnormalized_flux_errors)
        assert normalized_fluxes[10] == 0
        assert normalized_fluxes[90] == 0
        assert normalized_flux_errors[10] == 0
        assert normalized_flux_errors[90] == 0

    def test_make_uniform_length_does_not_change_input_that_is_already_the_correct_size(self, database):
        fluxes = np.array([0, 1, 2, 3, 4, 5])
        uniform_length_fluxes = database.make_uniform_length(fluxes, 6, randomize=False)
        assert np.array_equal(uniform_length_fluxes, [0, 1, 2, 3, 4, 5])

    def test_make_uniform_length_with_random_rolls_input_that_is_already_the_correct_size(self, database):
        fluxes = np.array([0, 1, 2, 3, 4, 5])
        with patch.object(module.np.random, 'randint') as stub_randint:
            stub_randint.return_value = 3
            uniform_length_fluxes = database.make_uniform_length(fluxes, 6, randomize=True)
            assert np.array_equal(uniform_length_fluxes, [3, 4, 5, 0, 1, 2])

    def test_make_uniform_length_repeats_elements_when_input_is_too_short(self, database):
        fluxes = np.array([0, 1, 2, 3])
        uniform_length_fluxes = database.make_uniform_length(fluxes, 6, randomize=False)
        assert np.array_equal(uniform_length_fluxes, [0, 1, 2, 3, 0, 1])

    def test_make_uniform_length_repeats_elements_when_input_is_too_short_with_random_roll(self, database):
        fluxes = np.array([0, 1, 2, 3])
        with patch.object(module.np.random, 'randint') as stub_randint:
            stub_randint.return_value = 3
            uniform_length_fluxes = database.make_uniform_length(fluxes, 6)
            assert np.array_equal(uniform_length_fluxes, [1, 2, 3, 0, 1, 2])

    def test_make_uniform_length_clips_elements_when_input_is_too_long(self, database):
        fluxes = np.array([0, 1, 2, 3, 4, 5])
        uniform_length_fluxes = database.make_uniform_length(fluxes, 4, randomize=False)
        assert np.array_equal(uniform_length_fluxes, [0, 1, 2, 3])

    def test_make_uniform_length_clips_elements_when_input_is_too_long_with_random_roll(self, database):
        fluxes = np.array([0, 1, 2, 3, 4, 5])
        with patch.object(module.np.random, 'randint') as stub_randint:
            stub_randint.return_value = 3
            uniform_length_fluxes = database.make_uniform_length(fluxes, 4, randomize=True)
            assert np.array_equal(uniform_length_fluxes, [3, 4, 5, 0])

    def test_make_uniform_length_repeats_elements_in_2d_array_when_input_is_too_short(self, database):
        fluxes = np.array([[0, 0], [1, -1]])
        uniform_length_fluxes = database.make_uniform_length(fluxes, 3, randomize=False)
        assert np.array_equal(uniform_length_fluxes, [[0, 0], [1, -1], [0, 0]])

    def test_remove_random_elements_removes_elements(self, database):
        array = np.array([0, 1, 2, 3])
        with patch.object(module.np.random, 'randint') as mock_randint:
            mock_randint.side_effect = lambda x: x
            updated_array = database.remove_random_elements(array, ratio=0.5)
        assert updated_array.shape[0] == 2

    def test_remove_random_elements_acts_on_axis_0(self, database):
        array = np.array([[0, 0], [1, -1], [2, -2], [3, -3]])
        with patch.object(module.np.random, 'choice') as mock_random_choice:
            mock_random_choice.return_value = [0, 2]
            updated_array = database.remove_random_elements(array)
        assert np.array_equal(updated_array, np.array([[1, -1], [3, -3]]))

    def test_can_normalize_the_flux_channel_of_a_light_curve(self):
        database = LightCurveDatabase()
        database.include_time_as_channel = True
        light_curve = np.array([[1, -1], [2, -2], [3, -3]])
        mock_normalize = Mock(side_effect=lambda x: x)
        database.normalize_on_percentiles = mock_normalize
        database.normalize_fluxes(light_curve=light_curve)
        assert np.array_equal(mock_normalize.call_args[0][0], light_curve[:, 1])  # Channel 1 should be fluxes.

    def test_can_normalize_the_flux_channel_of_a_flux_only_light_curve(self):
        database = LightCurveDatabase()
        light_curve = np.array([[1], [2], [3]])
        mock_normalize = Mock(side_effect=lambda x: x)
        database.normalize_on_percentiles = mock_normalize
        database.normalize_fluxes(light_curve=light_curve)
        assert np.array_equal(mock_normalize.call_args[0][0], light_curve[:, 0])  # Channel 1 should be fluxes.

    def test_flux_preprocessing_occurs_in_place(self):
        database = LightCurveDatabase()
        light_curve = np.array([[10], [20], [10], [20]])
        expected_light_curve = np.array([[-1], [1], [-1], [1]])
        database.normalize_fluxes(light_curve=light_curve)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_flux_preprocessing_with_times_and_errors(self):
        database = LightCurveDatabase()
        database.include_time_as_channel = True
        database.include_flux_errors_as_channel = True
        light_curve = np.array([[1, 10, 10], [2, 20, 20], [3, 10, 30], [4, 20, 40]])
        expected_light_curve = np.array([[1, -1, 2], [2, 1, 4], [3, -1, 6], [4, 1, 8]])
        database.normalize_fluxes(light_curve=light_curve)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_building_light_curve_array_using_only_fluxes(self):
        database = LightCurveDatabase()
        fluxes = np.array([1, 2, 3])
        expected_light_curve = np.array([[1], [2], [3]])
        light_curve = database.build_light_curve_array(fluxes=fluxes)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_building_light_curve_array_using_times_and_fluxes(self):
        database = LightCurveDatabase()
        database.include_time_as_channel = True
        times = np.array([1, 2, 3])
        fluxes = np.array([-1, -2, -3])
        expected_light_curve = np.array([[1, -1], [2, -2], [3, -3]])
        light_curve = database.build_light_curve_array(fluxes=fluxes, times=times)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_building_light_curve_array_passing_times_but_only_using_fluxes(self):
        database = LightCurveDatabase()
        database.include_time_as_channel = False
        times = np.array([1, 2, 3])
        fluxes = np.array([-1, -2, -3])
        expected_light_curve = np.array([[-1], [-2], [-3]])
        light_curve = database.build_light_curve_array(fluxes=fluxes, times=times)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_building_light_curve_array_passing_flux_errors_but_only_using_them(self):
        database = LightCurveDatabase()
        database.include_time_as_channel = False
        times = np.array([1, 2, 3])
        fluxes = np.array([-1, -2, -3])
        flux_errors = np.array([0.1, 0.2, 0.3])
        expected_light_curve = np.array([[-1], [-2], [-3]])
        light_curve = database.build_light_curve_array(fluxes=fluxes, times=times, flux_errors=flux_errors)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_building_light_curve_array_using_times_fluxes_and_flux_errors(self):
        database = LightCurveDatabase()
        database.include_time_as_channel = True
        database.include_flux_errors_as_channel = True
        times = np.array([1, 2, 3])
        fluxes = np.array([-1, -2, -3])
        flux_errors = np.array([0.1, 0.2, 0.3])
        expected_light_curve = np.array([[1, -1, 0.1], [2, -2, 0.2], [3, -3, 0.3]])
        light_curve = database.build_light_curve_array(fluxes=fluxes, times=times, flux_errors=flux_errors)
        assert np.array_equal(light_curve, expected_light_curve)

    def test_time_preprocessing_occurs_in_place(self):
        database = LightCurveDatabase()
        light_curve_array = np.array([[10, 1], [20, 2], [30, 3]])
        time_differences_result = [10, 10, 10]
        database.calculate_time_differences = Mock(return_value=time_differences_result)
        expected_light_curve_array = np.array([[10, 1], [10, 2], [10, 3]])
        database.preprocess_times(light_curve_array=light_curve_array)
        assert np.array_equal(light_curve_array, expected_light_curve_array)

    def test_time_normalization_occurs_in_place(self):
        database = LightCurveDatabase()
        times = np.array([10, 20, 30])
        expected_time_differences = np.array([10, 10, 10])
        time_differences = database.calculate_time_differences(times)
        assert np.array_equal(time_differences, expected_time_differences)

    @pytest.mark.parametrize("evaluation_mode, called_expectation", [(True, False),
                                                                     (False, True)])
    def test_flux_preprocessing_evaluation_modes_calling_of_remove_random_elements(self, database, evaluation_mode,
                                                                                   called_expectation):
        mock_remove_random_elements = Mock(side_effect=lambda x: x)
        database.remove_random_elements = mock_remove_random_elements

        database.preprocess_light_curve(np.array([[0], [1], [2]]), evaluation_mode=evaluation_mode)

        assert mock_remove_random_elements.called == called_expectation
