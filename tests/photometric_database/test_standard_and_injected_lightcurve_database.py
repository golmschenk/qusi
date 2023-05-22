from unittest.mock import patch, Mock

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

import ramjet.photometric_database.light_curve_database
import ramjet.photometric_database.standard_and_injected_light_curve_database as database_module
from ramjet.photometric_database.derived.toy_database import ToyRamjetDatabaseWithAuxiliary, ToyRamjetDatabaseWithFlatValueAsLabel
from ramjet.photometric_database.light_curve_collection import LightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import \
    StandardAndInjectedLightCurveDatabase, OutOfBoundsInjectionHandlingMethod
import ramjet.photometric_database.light_curve_database as light_curve_database_module


class TestStandardAndInjectedLightCurveDatabase:
    @pytest.fixture
    def database(self) -> StandardAndInjectedLightCurveDatabase:
        """A fixture of a blank database."""
        return StandardAndInjectedLightCurveDatabase()

    @pytest.fixture
    def database_with_collections(self) -> StandardAndInjectedLightCurveDatabase:
        """A fixture of the database with light_curve collections pre-prepared"""
        database = StandardAndInjectedLightCurveDatabase()
        # Setup mock light_curve collections.
        standard_light_curve_collection0 = LightCurveCollection()
        standard_light_curve_collection0.get_paths = lambda: [Path('standard_path0.ext')]
        standard_light_curve_collection0.load_times_and_fluxes_from_path = lambda path: (np.array([10, 20, 30]),
                                                                                         np.array([0, 1, 2]))
        standard_light_curve_collection0.label = 0
        standard_light_curve_collection1 = LightCurveCollection()
        standard_light_curve_collection1.get_paths = lambda: [Path('standard_path1.ext')]
        standard_light_curve_collection1.load_times_and_fluxes_from_path = lambda path: (np.array([20, 30, 40]),
                                                                                         np.array([1, 2, 3]))
        standard_light_curve_collection1.label = 1
        injectee_light_curve_collection = LightCurveCollection()
        injectee_light_curve_collection.get_paths = lambda: [Path('injectee_path.ext')]
        injectee_light_curve_collection.load_times_and_fluxes_from_path = lambda path: (np.array([30, 40, 50]),
                                                                                        np.array([2, 3, 4]))
        injectee_light_curve_collection.label = 0
        injectable_light_curve_collection0 = LightCurveCollection()
        injectable_light_curve_collection0.get_paths = lambda: [Path('injectable_path0.ext')]
        injectable_light_curve_collection0.load_times_and_magnifications_from_path = lambda path: (
            np.array([0, 10, 20]), np.array([0.5, 1, 1.5]))
        injectable_light_curve_collection0.label = 0
        injectable_light_curve_collection1 = LightCurveCollection()
        injectable_light_curve_collection1.get_paths = lambda: [Path('injectable_path1.ext')]
        injectable_light_curve_collection1.load_times_and_magnifications_from_path = lambda path: (
            np.array([0, 10, 20, 30]), np.array([0, 1, 1, 0]))
        injectable_light_curve_collection1.label = 1
        database.training_standard_light_curve_collections = [standard_light_curve_collection0,
                                                              standard_light_curve_collection1]
        database.training_injectee_light_curve_collection = injectee_light_curve_collection
        database.training_injectable_light_curve_collections = [injectable_light_curve_collection0,
                                                                injectable_light_curve_collection1]
        database.validation_standard_light_curve_collections = [standard_light_curve_collection1]
        database.validation_injectee_light_curve_collection = injectee_light_curve_collection
        database.validation_injectable_light_curve_collections = [injectable_light_curve_collection1]
        # Setup simplified database settings
        database.batch_size = 4
        database.time_steps_per_example = 3
        database.number_of_parallel_processes_per_map = 1

        def mock_window(dataset, batch_size, window_shift):
            return dataset.batch(batch_size)

        database.window_dataset_for_zipped_example_and_label_dataset = mock_window  # Disable windowing.
        database.normalize_on_percentiles = lambda fluxes: fluxes  # Don't normalize values to keep it simple.
        return database

    @pytest.fixture
    def deterministic_database(self, database_with_collections) -> StandardAndInjectedLightCurveDatabase:
        """A fixture of a deterministic database with light_curve collections pre-prepared."""
        database_with_collections.remove_random_elements = lambda x: x
        database_with_collections.randomly_roll_elements = lambda x: x
        return database_with_collections

    def test_database_has_light_curve_collection_properties(self):
        database = StandardAndInjectedLightCurveDatabase()
        assert hasattr(database, 'training_standard_light_curve_collections')
        assert hasattr(database, 'training_injectee_light_curve_collection')
        assert hasattr(database, 'training_injectable_light_curve_collections')
        assert hasattr(database, 'validation_standard_light_curve_collections')
        assert hasattr(database, 'validation_injectee_light_curve_collection')
        assert hasattr(database, 'validation_injectable_light_curve_collections')

    @pytest.mark.slow
    @pytest.mark.functional
    @patch.object(database_module.np.random, 'random', return_value=0)
    @patch.object(ramjet.photometric_database.light_curve_database.np.random, 'randint', return_value=0)
    def test_database_can_generate_training_and_validation_datasets(self, mock_randint, mock_random,
                                                                    database_with_collections):
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            training_dataset, validation_dataset = database_with_collections.generate_datasets()
        training_batch = next(iter(training_dataset))
        training_batch_examples = training_batch[0]
        training_batch_labels = training_batch[1]
        assert training_batch_examples.shape == (database_with_collections.batch_size, 3, 1)
        assert training_batch_labels.shape == (database_with_collections.batch_size, 1)
        assert np.array_equal(training_batch_examples[0].numpy(), [[0], [1], [2]])  # Standard light_curve 0.
        assert np.array_equal(training_batch_labels[0].numpy(), [0])  # Standard label 0.
        assert np.array_equal(training_batch_examples[1].numpy(), [[1], [2], [3]])  # Standard light_curve 1.
        assert np.array_equal(training_batch_labels[1].numpy(), [1])  # Standard label 1.
        assert np.array_equal(training_batch_examples[2].numpy(), [[0.5], [3], [5.5]])  # Injected light_curve 0.
        assert np.array_equal(training_batch_examples[3].numpy(), [[-1], [3], [4]])  # Injected light_curve 1.
        validation_batch = next(iter(validation_dataset))
        validation_batch_examples = validation_batch[0]
        validation_batch_labels = validation_batch[1]
        assert validation_batch_examples.shape == (database_with_collections.batch_size, 3, 1)
        assert validation_batch_labels.shape == (database_with_collections.batch_size, 1)
        assert np.array_equal(validation_batch_examples[0].numpy(), [[1], [2], [3]])  # Standard light_curve 1.
        assert np.array_equal(validation_batch_labels[0].numpy(), [1])  # Standard label 1.
        assert np.array_equal(validation_batch_examples[1].numpy(), [[-1], [3], [4]])  # Injected light_curve 1.
        assert np.array_equal(validation_batch_labels[1].numpy(), [1])  # Injected label 1.
        assert np.array_equal(validation_batch_examples[2].numpy(), [[1], [2], [3]])  # Standard light_curve 1.
        assert np.array_equal(validation_batch_examples[3].numpy(), [[-1], [3], [4]])  # Injected light_curve 1.

    @pytest.mark.slow
    @pytest.mark.functional
    def test_can_generate_standard_light_curve_and_label_dataset_from_paths_dataset_and_label(self,
                                                                                              deterministic_database):
        database = deterministic_database
        light_curve_collection = database.training_standard_light_curve_collections[0]
        paths_dataset = database.generate_paths_dataset_from_light_curve_collection(light_curve_collection)
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            light_curve_and_label_dataset = database.generate_standard_light_curve_and_label_dataset(
                paths_dataset,
                light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                light_curve_collection.load_auxiliary_information_for_path,
                light_curve_collection.load_label_from_path)
        light_curve_and_label = next(iter(light_curve_and_label_dataset))
        assert light_curve_and_label[0].numpy().shape == (3, 1)
        assert np.array_equal(light_curve_and_label[0].numpy(), [[0], [1], [2]])  # Standard light_curve 0.
        assert np.array_equal(light_curve_and_label[1].numpy(), [0])  # Standard label 0.

    def test_can_preprocess_standard_light_curve(self, deterministic_database):
        database = deterministic_database
        light_curve_collection = database.training_standard_light_curve_collections[0]
        # noinspection PyUnresolvedReferences
        light_curve_path = light_curve_collection.get_paths()[0]
        load_label_from_path_function = light_curve_collection.load_label_from_path
        expected_label = load_label_from_path_function(Path())
        load_from_path_function = light_curve_collection.load_times_fluxes_and_flux_errors_from_path
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            light_curve, label = database.preprocess_standard_light_curve(load_from_path_function,
                                                                          light_curve_collection.load_auxiliary_information_for_path,
                                                                          load_label_from_path_function,
                                                                          tf.convert_to_tensor(str(light_curve_path)))
        assert light_curve.shape == (3, 1)
        assert np.array_equal(light_curve, [[0], [1], [2]])  # Standard light_curve 0.
        assert np.array_equal(label, [expected_label])  # Standard label 0.

    def test_can_preprocess_standard_light_curve_with_passed_functions(self):
        database = StandardAndInjectedLightCurveDatabase()
        stub_load_times_fluxes_flux_errors_function = Mock(
            return_value=(np.array([0, -1, -2]), np.array([0, 1, 2]), None))
        mock_load_label_function = Mock(return_value=3)
        path_tensor = tf.constant('stub_path.fits')
        database.preprocess_light_curve = lambda identity, *args, **kwargs: identity

        # noinspection PyTypeChecker
        example, label = database.preprocess_standard_light_curve(
            load_times_fluxes_and_flux_errors_from_path_function=stub_load_times_fluxes_flux_errors_function,
            load_auxiliary_information_for_path_function=lambda path: np.array([], dtype=np.float32),
            load_label_from_path_function=mock_load_label_function, light_curve_path_tensor=path_tensor)

        assert np.array_equal(example, [[0], [1], [2]])
        assert np.array_equal(label, [3])

    def test_can_preprocess_injected_light_curve_with_passed_functions(self):
        database = StandardAndInjectedLightCurveDatabase()
        stub_load_function = Mock(return_value=(np.array([0, -1, -2]), np.array([0, 1, 2]), None))
        stub_auxiliary_information_function = Mock(return_value=np.array([], dtype=np.float32))
        mock_load_label_function = Mock(return_value=3)
        path_tensor = tf.constant('stub_path.fits')
        database.preprocess_light_curve = lambda identity, *args, **kwargs: identity
        database.inject_signal_into_light_curve = lambda identity, *args, **kwargs: identity

        # noinspection PyTypeChecker
        example, label = database.preprocess_injected_light_curve(
            injectee_load_times_fluxes_and_flux_errors_from_path_function=stub_load_function,
            load_auxiliary_information_for_path_function=stub_auxiliary_information_function,
            injectable_load_times_magnifications_and_magnification_errors_from_path_function=stub_load_function,
            load_label_from_path_function=mock_load_label_function, injectee_light_curve_path_tensor=path_tensor,
            injectable_light_curve_path_tensor=path_tensor)

        assert np.array_equal(example, [[0], [1], [2]])
        assert np.array_equal(label, [3])

    @pytest.mark.slow
    @pytest.mark.functional
    @patch.object(database_module.np.random, 'random', return_value=0)
    @patch.object(ramjet.photometric_database.light_curve_database.np.random, 'randint', return_value=0)
    def test_can_generate_injected_light_curve_and_label_dataset_from_paths_dataset_and_label(self, mock_randint,
                                                                                              mock_random,
                                                                                              database_with_collections):
        injectee_light_curve_collection = database_with_collections.training_injectee_light_curve_collection
        injectable_light_curve_collection = database_with_collections.training_injectable_light_curve_collections[0]
        injectee_paths_dataset = database_with_collections.generate_paths_dataset_from_light_curve_collection(
            injectee_light_curve_collection)
        injectable_paths_dataset = database_with_collections.generate_paths_dataset_from_light_curve_collection(
            injectable_light_curve_collection)
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            light_curve_and_label_dataset = database_with_collections.generate_injected_light_curve_and_label_dataset(
                injectee_paths_dataset, injectee_light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                injectee_light_curve_collection.load_auxiliary_information_for_path,
                injectable_paths_dataset,
                injectable_light_curve_collection.load_times_magnifications_and_magnification_errors_from_path,
                injectable_light_curve_collection.load_label_from_path)
        light_curve_and_label = next(iter(light_curve_and_label_dataset))
        assert light_curve_and_label[0].numpy().shape == (3, 1)
        assert np.array_equal(light_curve_and_label[0].numpy(), [[0.5], [3], [5.5]])  # Injected light_curve 0
        assert np.array_equal(light_curve_and_label[1].numpy(), [0])  # Injected label 0.

    def test_can_preprocess_injected_light_curve(self, deterministic_database):
        database = deterministic_database
        injectee_light_curve_collection = database.training_injectee_light_curve_collection
        injectable_light_curve_collection = database.training_injectable_light_curve_collections[0]
        # noinspection PyUnresolvedReferences
        injectee_light_curve_path = injectee_light_curve_collection.get_paths()[0]
        injectee_load_from_path_function = injectee_light_curve_collection.load_times_fluxes_and_flux_errors_from_path
        # noinspection PyUnresolvedReferences
        injectable_light_curve_path = injectable_light_curve_collection.get_paths()[0]
        load_label_from_path_function = injectable_light_curve_collection.load_label_from_path
        expected_label = load_label_from_path_function(Path())
        injectable_load_from_path_function = \
            injectable_light_curve_collection.load_times_magnifications_and_magnification_errors_from_path
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            light_curve, label = database.preprocess_injected_light_curve(
                injectee_load_from_path_function,
                injectee_light_curve_collection.load_auxiliary_information_for_path,
                injectable_load_from_path_function,
                load_label_from_path_function,
                tf.convert_to_tensor(str(injectee_light_curve_path)),
                tf.convert_to_tensor(str(injectable_light_curve_path)))
        assert light_curve.shape == (3, 1)
        assert np.array_equal(light_curve, [[0.5], [3], [5.5]])  # Injected light_curve 0.
        assert np.array_equal(label, [expected_label])  # Injected label 0.

    def test_can_create_tensorflow_dataset_for_light_curve_collection_paths(self, database_with_collections):
        injectee_paths_dataset = database_with_collections.generate_paths_dataset_from_light_curve_collection(
            database_with_collections.training_injectee_light_curve_collection)
        assert next(iter(injectee_paths_dataset)).numpy() == b'injectee_path.ext'

    def test_light_curve_collection_paths_dataset_is_repeated(self, database_with_collections):
        with patch.object(database_module.tf.data.Dataset, 'repeat',
                          side_effect=lambda dataset: dataset, autospec=True) as mock_repeat:
            _ = database_with_collections.generate_paths_dataset_from_light_curve_collection(
                database_with_collections.training_injectee_light_curve_collection)
            assert mock_repeat.called

    def test_light_curve_collection_paths_dataset_is_shuffled(self, database_with_collections):
        with patch.object(database_module.tf.data.Dataset, 'shuffle',
                          side_effect=lambda dataset, buffer_size: dataset, autospec=True) as mock_shuffle:
            _ = database_with_collections.generate_paths_dataset_from_light_curve_collection(
                database_with_collections.training_injectee_light_curve_collection)
            assert mock_shuffle.called
            assert mock_shuffle.call_args[0][1] == database_with_collections.shuffle_buffer_size

    def test_can_create_tensorflow_datasets_for_multiple_light_curve_collections_paths(self, database_with_collections):
        standard_paths_datasets = database_with_collections.generate_paths_datasets_from_light_curve_collection_list(
            database_with_collections.training_standard_light_curve_collections)
        assert next(iter(standard_paths_datasets[0])).numpy() == b'standard_path0.ext'
        assert next(iter(standard_paths_datasets[1])).numpy() == b'standard_path1.ext'

    def test_can_inject_signal_into_fluxes(self):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5])
        light_curve_times = np.array([10, 20, 30, 40, 50])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database_module.inject_signal_into_light_curve(
            light_curve_times,
            light_curve_fluxes,
            signal_times,
            signal_magnifications)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5]))

    def test_inject_signal_errors_on_out_of_bounds(self):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with pytest.raises(ValueError):
            database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications)

    def test_inject_signal_can_be_told_to_allow_out_of_bounds(self):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            fluxes_with_injected_signal = database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))

    def test_inject_signal_using_repeats_for_out_of_bounds(self):
        light_curve_fluxes = np.array([1, 1, 1, 1, 1, 1, 1])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60, 70])
        signal_magnifications = np.array([1, 2])
        signal_times = np.array([0, 10])
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0.6  # Make signal offset end up as 40
            fluxes_with_injected_signal0 = database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL)
        assert np.array_equal(fluxes_with_injected_signal0, np.array([2, 1, 2, 1, 2, 1, 2]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0.8  # Make signal offset end up as 50
            fluxes_with_injected_signal1 = database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL)
        assert np.array_equal(fluxes_with_injected_signal1, np.array([1, 2, 1, 2, 1, 2, 1]))

    def test_injected_signal_randomly_varies_injectable_portion_used_when_injectable_larger_than_injectee(
            self):
        injectee_fluxes = np.array([1, 2, 3])
        injectee_times = np.array([10, 20, 30])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications)
            assert np.array_equal(injected, np.array([1, 4, 7]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 1
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications)
            assert np.array_equal(injected, np.array([5, 4, 3]))

    def test_injected_signal_randomly_varies_injection_location_when_injectee_larger_than_injectable(
            self):
        injectee_fluxes = np.array([1, 2, 3, 4, 5])
        injectee_times = np.array([10, 20, 30, 40, 50])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 10, 20])
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION)
            assert np.array_equal(injected, np.array([1, 8, 3, 4, 5]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 1
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION)
            assert np.array_equal(injected, np.array([1, 2, 3, 10, 5]))

    def test_database_can_inject_signal_into_fluxes(self, database_with_collections):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5])
        light_curve_times = np.array([10, 20, 30, 40, 50])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database_with_collections.inject_signal_into_light_curve(light_curve_fluxes,
                                                                                               light_curve_times,
                                                                                               signal_magnifications,
                                                                                               signal_times)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5]))

    def test_database_inject_signal_errors_on_out_of_bounds(self, database_with_collections):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with pytest.raises(ValueError):
            database_with_collections.inject_signal_into_light_curve(light_curve_fluxes, light_curve_times,
                                                                     signal_magnifications, signal_times)

    def test_database_inject_signal_can_be_told_to_allow_out_of_bounds(self, database_with_collections):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        database_with_collections.out_of_bounds_injection_handling = \
            OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            fluxes_with_injected_signal = database_with_collections.inject_signal_into_light_curve(light_curve_fluxes,
                                                                                                   light_curve_times,
                                                                                                   signal_magnifications,
                                                                                                   signal_times)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))

    def test_database_inject_signal_using_repeats_for_out_of_bounds(self, database_with_collections):
        light_curve_fluxes = np.array([1, 1, 1, 1, 1, 1, 1])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60, 70])
        signal_magnifications = np.array([1, 2])
        signal_times = np.array([0, 10])
        database_with_collections.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0.6  # Make signal offset end up as 40
            fluxes_with_injected_signal0 = database_with_collections.inject_signal_into_light_curve(light_curve_fluxes,
                                                                                                    light_curve_times,
                                                                                                    signal_magnifications,
                                                                                                    signal_times)
        assert np.array_equal(fluxes_with_injected_signal0, np.array([2, 1, 2, 1, 2, 1, 2]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0.8  # Make signal offset end up as 50
            fluxes_with_injected_signal1 = database_with_collections.inject_signal_into_light_curve(light_curve_fluxes,
                                                                                                    light_curve_times,
                                                                                                    signal_magnifications,
                                                                                                    signal_times)
        assert np.array_equal(fluxes_with_injected_signal1, np.array([1, 2, 1, 2, 1, 2, 1]))

    def test_database_injected_signal_randomly_varies_injectable_portion_used_when_injectable_larger_than_injectee(
            self, database_with_collections):
        injectee_fluxes = np.array([1, 2, 3])
        injectee_times = np.array([10, 20, 30])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            injected = database_with_collections.inject_signal_into_light_curve(injectee_fluxes, injectee_times,
                                                                                injectable_magnifications,
                                                                                injectable_times)
            assert np.array_equal(injected, np.array([1, 4, 7]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 1
            injected = database_with_collections.inject_signal_into_light_curve(injectee_fluxes, injectee_times,
                                                                                injectable_magnifications,
                                                                                injectable_times)
            assert np.array_equal(injected, np.array([5, 4, 3]))

    def test_database_injected_signal_randomly_varies_injection_location_when_injectee_larger_than_injectable(
            self, database_with_collections):
        injectee_fluxes = np.array([1, 2, 3, 4, 5])
        injectee_times = np.array([10, 20, 30, 40, 50])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 10, 20])
        database_with_collections.out_of_bounds_injection_handling = \
            OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            injected = database_with_collections.inject_signal_into_light_curve(injectee_fluxes, injectee_times,
                                                                                injectable_magnifications,
                                                                                injectable_times)
            assert np.array_equal(injected, np.array([1, 8, 3, 4, 5]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 1
            injected = database_with_collections.inject_signal_into_light_curve(injectee_fluxes, injectee_times,
                                                                                injectable_magnifications,
                                                                                injectable_times)
            assert np.array_equal(injected, np.array([1, 2, 3, 10, 5]))

    def test_can_intersperse_datasets(self, database_with_collections):
        dataset0 = tf.data.Dataset.from_tensor_slices([[0], [2], [4]])
        dataset1 = tf.data.Dataset.from_tensor_slices([[1], [3], [5]])
        interspersed_dataset = database_module.intersperse_datasets([dataset0, dataset1])
        assert list(interspersed_dataset) == [[0], [1], [2], [3], [4], [5]]

    def test_can_intersperse_zipped_example_label_datasets(self, database_with_collections):
        examples_dataset0 = tf.data.Dataset.from_tensor_slices([[0, 0], [2, 2], [4, 4]])
        labels_dataset0 = tf.data.Dataset.from_tensor_slices([[0], [-2], [-4]])
        dataset0 = tf.data.Dataset.zip((examples_dataset0, labels_dataset0))
        examples_dataset1 = tf.data.Dataset.from_tensor_slices([[1, 1], [3, 3], [5, 5]])
        labels_dataset1 = tf.data.Dataset.from_tensor_slices([[-1], [-3], [-5]])
        dataset1 = tf.data.Dataset.zip((examples_dataset1, labels_dataset1))
        interspersed_dataset = database_module.intersperse_datasets([dataset0, dataset1])
        interspersed_dataset_iterator = iter(interspersed_dataset)
        examples_and_labels0 = next(interspersed_dataset_iterator)
        assert np.array_equal(examples_and_labels0[0], [0, 0])
        assert examples_and_labels0[1] == [0]
        examples_and_labels1 = next(interspersed_dataset_iterator)
        assert np.array_equal(examples_and_labels1[0], [1, 1])
        assert examples_and_labels1[1] == [-1]
        examples_and_labels2 = next(interspersed_dataset_iterator)
        assert np.array_equal(examples_and_labels2[0], [2, 2])
        assert examples_and_labels2[1] == [-2]

    @pytest.mark.slow
    @pytest.mark.functional
    @patch.object(ramjet.photometric_database.light_curve_database.np.random, 'randint', return_value=0)
    def test_database_can_generate_training_and_validation_datasets_with_only_standard_collections(
            self, mock_randint, database_with_collections):
        database_with_collections.training_injectee_light_curve_collection = None
        database_with_collections.training_injectable_light_curve_collections = []
        database_with_collections.validation_injectee_light_curve_collection = None
        database_with_collections.validation_injectable_light_curve_collections = []
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            training_dataset, validation_dataset = database_with_collections.generate_datasets()
        training_batch = next(iter(training_dataset))
        training_batch_examples = training_batch[0]
        training_batch_labels = training_batch[1]
        assert training_batch_examples.shape == (database_with_collections.batch_size, 3, 1)
        assert training_batch_labels.shape == (database_with_collections.batch_size, 1)
        assert np.array_equal(training_batch_examples[0].numpy(), [[0], [1], [2]])  # Standard light_curve 0.
        assert np.array_equal(training_batch_labels[0].numpy(), [0])  # Standard label 0.
        assert np.array_equal(training_batch_examples[1].numpy(), [[1], [2], [3]])  # Standard light_curve 1.
        assert np.array_equal(training_batch_labels[1].numpy(), [1])  # Standard label 1.
        assert np.array_equal(training_batch_examples[2].numpy(), [[0], [1], [2]])  # Standard light_curve 0.
        assert np.array_equal(training_batch_examples[3].numpy(), [[1], [2], [3]])  # Standard light_curve 1.

    @pytest.mark.slow
    @pytest.mark.functional
    def test_can_generate_infer_path_and_light_curve_dataset_from_paths_dataset_and_label(
            self, database_with_collections):
        light_curve_collection = database_with_collections.training_standard_light_curve_collections[0]
        paths_dataset = database_with_collections.generate_paths_dataset_from_light_curve_collection(
            light_curve_collection)
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            path_and_light_curve_dataset = database_with_collections.generate_infer_path_and_light_curve_dataset(
                paths_dataset, light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
                light_curve_collection.load_auxiliary_information_for_path)
        path_and_light_curve = next(iter(path_and_light_curve_dataset))
        assert np.array_equal(path_and_light_curve[0].numpy(), b'standard_path0.ext')  # Standard path 0.
        assert path_and_light_curve[1].numpy().shape == (3, 1)
        assert np.array_equal(path_and_light_curve[1].numpy(), [[0], [1], [2]])  # Standard light_curve 0.

    def test_can_preprocess_infer_light_curve(self, database_with_collections):
        light_curve_collection = database_with_collections.training_standard_light_curve_collections[0]
        # noinspection PyUnresolvedReferences
        light_curve_path = light_curve_collection.get_paths()[0]
        expected_label = light_curve_collection.label
        load_from_path_function = light_curve_collection.load_times_fluxes_and_flux_errors_from_path
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            path, light_curve = database_with_collections.preprocess_infer_light_curve(
                load_from_path_function,
                light_curve_collection.load_auxiliary_information_for_path,
                tf.convert_to_tensor(str(light_curve_path)))
        assert np.array_equal(path, 'standard_path0.ext')  # Standard path 0.
        assert light_curve.shape == (3, 1)
        assert np.array_equal(light_curve, [[0], [1], [2]])  # Standard light_curve 0.

    @pytest.mark.slow
    @pytest.mark.functional
    def test_generated_standard_and_infer_datasets_return_the_same_light_curve(self, database_with_collections):
        light_curve_collection = database_with_collections.training_standard_light_curve_collections[0]
        paths_dataset0 = database_with_collections.generate_paths_dataset_from_light_curve_collection(
            light_curve_collection)
        light_curve_and_label_dataset = database_with_collections.generate_standard_light_curve_and_label_dataset(
            paths_dataset0, light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
            light_curve_collection.load_auxiliary_information_for_path,
            light_curve_collection.load_label_from_path, evaluation_mode=True)
        light_curve_and_label = next(iter(light_curve_and_label_dataset))
        paths_dataset1 = database_with_collections.generate_paths_dataset_from_light_curve_collection(
            light_curve_collection)
        path_and_light_curve_dataset = database_with_collections.generate_infer_path_and_light_curve_dataset(
            paths_dataset1, light_curve_collection.load_times_fluxes_and_flux_errors_from_path,
            light_curve_collection.load_auxiliary_information_for_path)
        path_and_light_curve = next(iter(path_and_light_curve_dataset))
        assert np.array_equal(light_curve_and_label[0].numpy(), path_and_light_curve[1].numpy())

    @pytest.mark.functional
    def test_can_specify_a_label_with_more_then_size_one_in_preprocessor(self):
        database = StandardAndInjectedLightCurveDatabase()
        database.number_of_parallel_processes_per_map = 1
        database.time_steps_per_example = 3
        database.number_of_label_values = 2
        stub_load_times_fluxes_and_flux_errors = lambda path: (np.array([0, -1, -2]), np.array([0, 1, 2]), None)
        expected_label = np.array([0, 1])
        stub_load_label_function = lambda path: expected_label
        stub_load_auxiliary_data_function = lambda path: np.array([], dtype=np.float32)
        paths_dataset = tf.data.Dataset.from_tensor_slices(['a.fits', 'b.fits'])
        dataset = database.generate_standard_light_curve_and_label_dataset(paths_dataset,
                                                                           stub_load_times_fluxes_and_flux_errors,
                                                                           stub_load_auxiliary_data_function,
                                                                           stub_load_label_function)
        dataset_list = list(dataset)
        assert np.array_equal(dataset_list[0][1], expected_label)

    @pytest.mark.parametrize('original_label, expected_label', [(0, np.array([0])),
                                                                ([0], np.array([0])),
                                                                (np.array([0]), np.array([0])),
                                                                ([0, 0], np.array([0, 0]))])
    def test_expand_label_to_training_dimensions(self, original_label, expected_label):
        database = StandardAndInjectedLightCurveDatabase()
        label = database_module.expand_label_to_training_dimensions(original_label)
        assert type(label) is np.ndarray
        assert np.array_equal(label, expected_label)

    def test_grouping_from_light_curve_auxiliary_and_label_to_observation_and_label(self):
        light_curve_dataset = tf.data.Dataset.from_tensor_slices([[0, 0], [2, 2], [4, 4]])
        auxiliary_dataset = tf.data.Dataset.from_tensor_slices([[0], [20], [40]])
        label_dataset = tf.data.Dataset.from_tensor_slices([[0], [-2], [-4]])
        light_curve_auxiliary_and_label_dataset = tf.data.Dataset.zip(
            (light_curve_dataset, auxiliary_dataset, label_dataset))
        observation_and_label_dataset = \
            database_module.from_light_curve_auxiliary_and_label_to_observation_and_label(
                light_curve_auxiliary_and_label_dataset)
        dataset_iterator = iter(observation_and_label_dataset)
        observation_and_label0 = next(dataset_iterator)
        assert np.array_equal(observation_and_label0[0][0], [0, 0])
        assert np.array_equal(observation_and_label0[0][1], [0])
        assert np.array_equal(observation_and_label0[1], [0])
        observation_and_label1 = next(dataset_iterator)
        assert np.array_equal(observation_and_label1[0][0], [2, 2])
        assert np.array_equal(observation_and_label1[0][1], [20])
        assert np.array_equal(observation_and_label1[1], [-2])

    @pytest.mark.slow
    @pytest.mark.integration
    def test_database_can_generate_training_and_validation_datasets_with_auxiliary_input(self):
        database = StandardAndInjectedLightCurveDatabase()
        light_curve_collection0 = LightCurveCollection()
        light_curve_collection0.get_paths = lambda: [Path('path0.ext')]
        light_curve_collection0.load_times_and_fluxes_from_path = lambda path: (np.array([90, 100, 110]),
                                                                                np.array([0, 1, 2]))
        light_curve_collection0.label = 0
        light_curve_collection0.load_auxiliary_information_for_path = lambda path: np.array([3, 4])
        database.training_standard_light_curve_collections = [light_curve_collection0]
        database.validation_standard_light_curve_collections = [light_curve_collection0]
        database.remove_random_elements = lambda x: x  # Don't randomize values to keep it simple.
        database.randomly_roll_elements = lambda x: x  # Don't randomize values to keep it simple.
        database.normalize_on_percentiles = lambda fluxes: fluxes  # Don't normalize values to keep it simple.
        database.batch_size = 4
        database.time_steps_per_example = 3
        database.number_of_parallel_processes_per_map = 1
        database.number_of_auxiliary_values = 2
        with (
            patch.object(light_curve_database_module, 'normalize_on_percentiles') as mock_normalize_on_percentiles,
            patch.object(light_curve_database_module, 'randomly_roll_elements') as mock_randomly_roll_elements,
            patch.object(light_curve_database_module, 'remove_random_elements') as mock_remove_random_elements
        ):
            # Remove other preprocessing to keep it simple.
            mock_normalize_on_percentiles.side_effect = lambda fluxes: fluxes
            mock_randomly_roll_elements.side_effect = lambda fluxes: fluxes
            mock_remove_random_elements.side_effect = lambda fluxes: fluxes
            training_dataset, validation_dataset = database.generate_datasets()
        training_batch = next(iter(training_dataset))
        training_batch_observations = training_batch[0]
        training_batch_labels = training_batch[1]
        assert training_batch_observations[0].shape == (database.batch_size, 3, 1)
        assert training_batch_observations[1].shape == (database.batch_size, 2)
        assert training_batch_labels.shape == (database.batch_size, 1)
        assert np.array_equal(training_batch_observations[0][0].numpy(), [[0], [1], [2]])  # Light curve
        assert np.array_equal(training_batch_observations[1][0].numpy(), [3, 4])  # Auxiliary
        assert np.array_equal(training_batch_labels[0].numpy(), [0])  # Label.
        validation_batch = next(iter(validation_dataset))
        validation_batch_observations = validation_batch[0]
        validation_batch_labels = validation_batch[1]
        assert np.array_equal(validation_batch_observations[0][0].numpy(), [[0], [1], [2]])  # Light curve
        assert np.array_equal(validation_batch_observations[1][0].numpy(), [3, 4])  # Auxiliary
        assert np.array_equal(validation_batch_labels[0].numpy(), [0])  # Label.

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
        padded_window_dataset = database_module.padded_window_dataset_for_zipped_example_and_label_dataset(
            dataset=dataset, batch_size=3, window_shift=2, padded_shapes=([None], [None]))
        padded_window_iterator = iter(padded_window_dataset)
        batch0 = next(padded_window_iterator)
        assert np.array_equal(batch0[0].numpy(), [[1, 1], [2, 2], [3, 3]])
        batch1 = next(padded_window_iterator)
        assert np.array_equal(batch1[0].numpy(), [[3, 3, 0], [4, 4, 4], [5, 5, 5]])

    def test_window_dataset_for_zipped_example_and_label_dataset_produces_windowed_batches(self, database):
        example_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
        label_dataset = tf.data.Dataset.from_tensor_slices([-1, -2, -3, -4, -5])
        dataset = tf.data.Dataset.zip((example_dataset, label_dataset))
        windowed_dataset = database_module.window_dataset_for_zipped_example_and_label_dataset(dataset,
                                                                                               batch_size=3,
                                                                                               window_shift=2)
        windowed_dataset_iterator = iter(windowed_dataset)
        batch0 = next(windowed_dataset_iterator)
        assert np.array_equal(batch0[0], [1, 2, 3])
        assert np.array_equal(batch0[1], [-1, -2, -3])
        batch1 = next(windowed_dataset_iterator)
        assert np.array_equal(batch1[0], [3, 4, 5])
        assert np.array_equal(batch1[1], [-3, -4, -5])

    def test_flat_window_zipped_produces_overlapping_window_repeats(self, database):
        examples_dataset = tf.data.Dataset.from_tensor_slices(['a', 'b', 'c', 'd', 'e'])
        labels_dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4])
        zipped_dataset = tf.data.Dataset.zip((examples_dataset, labels_dataset))

        windowed_dataset = database_module.flat_window_zipped_example_and_label_dataset(zipped_dataset, batch_size=3,
                                                                                        window_shift=2)

        windowed_list = list(windowed_dataset.as_numpy_iterator())
        assert windowed_list == [(b'a', 0), (b'b', 1), (b'c', 2), (b'c', 2), (b'd', 3), (b'e', 4), (b'e', 4)]

    def test_flat_window_zipped_with_shuffle_keeps_correct_pairings(self, database):
        examples_dataset = tf.data.Dataset.from_tensor_slices(['a', 'b', 'c', 'd', 'e'])
        labels_dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4])
        zipped_dataset = tf.data.Dataset.zip((examples_dataset, labels_dataset))
        shuffled_zipped_dataset = zipped_dataset.shuffle(buffer_size=5)

        windowed_dataset = database_module.flat_window_zipped_example_and_label_dataset(shuffled_zipped_dataset,
                                                                                        batch_size=3,
                                                                                        window_shift=2)

        windowed_list = list(windowed_dataset.as_numpy_iterator())
        correct_pairings = {b'a': 0, b'b': 1, b'c': 2, b'd': 3, b'e': 4}
        for string, number in windowed_list:
            correct_number = correct_pairings[string]
            assert number == correct_number

    @pytest.mark.integration
    def test_labels_match_observation(self):
        database = ToyRamjetDatabaseWithAuxiliary()
        train_dataset, validation_dataset = database.generate_datasets()
        train_batch = next(iter(train_dataset))
        for light_curve_tensor, auxiliary_value_tensor, label_tensor in zip(
                train_batch[0][0], train_batch[0][1], train_batch[1]):
            light_curve_array = light_curve_tensor.numpy()
            if light_curve_array.min() == light_curve_array.max():
                assert label_tensor.numpy() == 0  # Flat signals should be label 0.
            else:
                assert label_tensor.numpy() == 1  # Sine wave signals should be label 1.

    @pytest.mark.integration
    def test_labels_match_observation_when_single_collection_has_multiple_labels(self):
        database = ToyRamjetDatabaseWithFlatValueAsLabel()
        train_dataset, validation_dataset = database.generate_datasets()
        train_batch = next(iter(train_dataset))
        for light_curve_tensor, label_tensor in zip(train_batch[0], train_batch[1]):
            assert label_tensor.numpy() == light_curve_tensor.numpy()[0]
