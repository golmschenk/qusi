from unittest.mock import patch, Mock

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

import ramjet.photometric_database.standard_and_injected_lightcurve_database as database_module
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import \
    StandardAndInjectedLightcurveDatabase, OutOfBoundsInjectionHandlingMethod


class TestStandardAndInjectedLightcurveDatabase:
    @pytest.fixture
    def database(self) -> StandardAndInjectedLightcurveDatabase:
        """A fixture of the database with lightcurve collections pre-prepared"""
        database = StandardAndInjectedLightcurveDatabase()
        # Setup mock lightcurve collections.
        standard_lightcurve_collection0 = LightcurveCollection(
            function_to_get_paths=lambda: [Path('standard_path0.ext')],
            function_to_load_times_and_fluxes_from_path=lambda path: (np.array([10, 20, 30]), np.array([0, 1, 2])),
            label=0)
        standard_lightcurve_collection1 = LightcurveCollection(
            function_to_get_paths=lambda: [Path('standard_path1.ext')],
            function_to_load_times_and_fluxes_from_path=lambda path: (np.array([20, 30, 40]), np.array([1, 2, 3])),
            label=1)
        injectee_lightcurve_collection = LightcurveCollection(
            function_to_get_paths=lambda: [Path('injectee_path.ext')],
            function_to_load_times_and_fluxes_from_path=lambda path: (np.array([30, 40, 50]), np.array([2, 3, 4])),
            label=0)
        injectable_lightcurve_collection0 = LightcurveCollection(
            function_to_get_paths=lambda: [Path('injectable_path0.ext')],
            function_to_load_times_and_magnifications_from_path=lambda path: (np.array([0, 10, 20]),
                                                                              np.array([0.5, 1, 1.5])),
            label=0)
        injectable_lightcurve_collection1 = LightcurveCollection(
            function_to_get_paths=lambda: [Path('injectable_path1.ext')],
            function_to_load_times_and_magnifications_from_path=lambda path: (np.array([0, 10, 20, 30]),
                                                                              np.array([0, 1, 1, 0])),
            label=1)
        database.training_standard_lightcurve_collections = [standard_lightcurve_collection0,
                                                             standard_lightcurve_collection1]
        database.training_injectee_lightcurve_collection = injectee_lightcurve_collection
        database.training_injectable_lightcurve_collections = [injectable_lightcurve_collection0,
                                                               injectable_lightcurve_collection1]
        database.validation_standard_lightcurve_collections = [standard_lightcurve_collection1]
        database.validation_injectee_lightcurve_collection = injectee_lightcurve_collection
        database.validation_injectable_lightcurve_collections = [injectable_lightcurve_collection1]
        # Setup simplified database settings
        database.batch_size = 4
        database.time_steps_per_example = 3
        database.number_of_parallel_processes_per_map = 1

        def mock_window(dataset, batch_size, window_shift):
            return dataset.batch(batch_size)

        database.window_dataset_for_zipped_example_and_label_dataset = mock_window  # Disable windowing.
        database.normalize = lambda fluxes: fluxes  # Don't normalize values to keep it simple.
        return database

    def test_database_has_lightcurve_collection_properties(self):
        database = StandardAndInjectedLightcurveDatabase()
        assert hasattr(database, 'training_standard_lightcurve_collections')
        assert hasattr(database, 'training_injectee_lightcurve_collection')
        assert hasattr(database, 'training_injectable_lightcurve_collections')
        assert hasattr(database, 'validation_standard_lightcurve_collections')
        assert hasattr(database, 'validation_injectee_lightcurve_collection')
        assert hasattr(database, 'validation_injectable_lightcurve_collections')

    @pytest.mark.slow
    @pytest.mark.functional
    @patch.object(database_module.np.random, 'random', return_value=0)
    def test_database_can_generate_training_and_validation_datasets(self, mock_random, database):
        training_dataset, validation_dataset = database.generate_datasets()
        training_batch = next(iter(training_dataset))
        training_batch_examples = training_batch[0]
        training_batch_labels = training_batch[1]
        assert training_batch_examples.shape == (database.batch_size, 3, 1)
        assert training_batch_labels.shape == (database.batch_size, 1)
        assert np.array_equal(training_batch_examples[0].numpy(), [[0], [1], [2]])  # Standard lightcurve 0.
        assert np.array_equal(training_batch_labels[0].numpy(), [0])  # Standard label 0.
        assert np.array_equal(training_batch_examples[1].numpy(), [[1], [2], [3]])  # Standard lightcurve 1.
        assert np.array_equal(training_batch_labels[1].numpy(), [1])  # Standard label 1.
        assert np.array_equal(training_batch_examples[2].numpy(), [[0.5], [3], [5.5]])  # Injected lightcurve 0.
        assert np.array_equal(training_batch_examples[3].numpy(), [[-1], [3], [4]])  # Injected lightcurve 1.
        validation_batch = next(iter(validation_dataset))
        validation_batch_examples = validation_batch[0]
        validation_batch_labels = validation_batch[1]
        assert validation_batch_examples.shape == (database.batch_size, 3, 1)
        assert validation_batch_labels.shape == (database.batch_size, 1)
        assert np.array_equal(validation_batch_examples[0].numpy(), [[1], [2], [3]])  # Standard lightcurve 1.
        assert np.array_equal(validation_batch_labels[0].numpy(), [1])  # Standard label 1.
        assert np.array_equal(validation_batch_examples[1].numpy(), [[-1], [3], [4]])  # Injected lightcurve 1.
        assert np.array_equal(validation_batch_labels[1].numpy(), [1])  # Injected label 1.
        assert np.array_equal(validation_batch_examples[2].numpy(), [[1], [2], [3]])  # Standard lightcurve 1.
        assert np.array_equal(validation_batch_examples[3].numpy(), [[-1], [3], [4]])  # Injected lightcurve 1.

    @pytest.mark.slow
    @pytest.mark.functional
    def test_can_generate_standard_lightcurve_and_label_dataset_from_paths_dataset_and_label(self, database):
        lightcurve_collection = database.training_standard_lightcurve_collections[0]
        paths_dataset = database.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
        lightcurve_and_label_dataset = database.generate_standard_lightcurve_and_label_dataset(
            paths_dataset, lightcurve_collection.load_times_and_fluxes_from_path,
            lightcurve_collection.load_label_from_path)
        lightcurve_and_label = next(iter(lightcurve_and_label_dataset))
        assert lightcurve_and_label[0].numpy().shape == (3, 1)
        assert np.array_equal(lightcurve_and_label[0].numpy(), [[0], [1], [2]])  # Standard lightcurve 0.
        assert np.array_equal(lightcurve_and_label[1].numpy(), [0])  # Standard label 0.

    def test_can_preprocess_standard_lightcurve(self, database):
        lightcurve_collection = database.training_standard_lightcurve_collections[0]
        # noinspection PyUnresolvedReferences
        lightcurve_path = lightcurve_collection.get_paths()[0]
        load_label_from_path_function = lightcurve_collection.load_label_from_path
        expected_label = load_label_from_path_function(Path())
        load_from_path_function = lightcurve_collection.load_times_and_fluxes_from_path
        lightcurve, label = database.preprocess_standard_lightcurve(load_from_path_function,
                                                                    load_label_from_path_function,
                                                                    tf.convert_to_tensor(str(lightcurve_path)))
        assert lightcurve.shape == (3, 1)
        assert np.array_equal(lightcurve, [[0], [1], [2]])  # Standard lightcurve 0.
        assert np.array_equal(label, [expected_label])  # Standard label 0.

    def test_can_preprocess_standard_lightcurve_with_passed_functions(self):
        database = StandardAndInjectedLightcurveDatabase()
        stub_load_times_and_fluxes_function = Mock(return_value=(np.array([0, -1, -2]), np.array([0, 1, 2])))
        mock_load_label_function = Mock(return_value=3)
        path_tensor = tf.constant('stub_path.fits')
        database.flux_preprocessing = lambda identity: identity

        # noinspection PyTypeChecker
        example, label = database.preprocess_standard_lightcurve(
            load_times_and_fluxes_from_path_function=stub_load_times_and_fluxes_function,
            load_label_from_path_function=mock_load_label_function,
            lightcurve_path_tensor=path_tensor
        )

        assert np.array_equal(example, [[0], [1], [2]])
        assert np.array_equal(label, [3])

    def test_can_preprocess_injected_lightcurve_with_passed_functions(self):
        database = StandardAndInjectedLightcurveDatabase()
        stub_load_times_and_fluxes_function = Mock(return_value=(np.array([0, -1, -2]), np.array([0, 1, 2])))
        mock_load_label_function = Mock(return_value=3)
        path_tensor = tf.constant('stub_path.fits')
        database.flux_preprocessing = lambda identity: identity
        database.inject_signal_into_lightcurve = lambda identity, *other_args: identity

        # noinspection PyTypeChecker
        example, label = database.preprocess_injected_lightcurve(
            injectee_load_times_and_fluxes_from_path_function=stub_load_times_and_fluxes_function,
            injectable_load_times_and_magnifications_from_path_function=stub_load_times_and_fluxes_function,
            load_label_from_path_function=mock_load_label_function,
            injectable_lightcurve_path_tensor=path_tensor,
            injectee_lightcurve_path_tensor=path_tensor
        )

        assert np.array_equal(example, [[0], [1], [2]])
        assert np.array_equal(label, [3])

    @pytest.mark.slow
    @pytest.mark.functional
    @patch.object(database_module.np.random, 'random', return_value=0)
    def test_can_generate_injected_lightcurve_and_label_dataset_from_paths_dataset_and_label(self, mock_random,
                                                                                             database):
        injectee_lightcurve_collection = database.training_injectee_lightcurve_collection
        injectable_lightcurve_collection = database.training_injectable_lightcurve_collections[0]
        injectee_paths_dataset = database.generate_paths_dataset_from_lightcurve_collection(
            injectee_lightcurve_collection)
        injectable_paths_dataset = database.generate_paths_dataset_from_lightcurve_collection(
            injectable_lightcurve_collection)
        lightcurve_and_label_dataset = database.generate_injected_lightcurve_and_label_dataset(
            injectee_paths_dataset, injectee_lightcurve_collection.load_times_and_fluxes_from_path,
            injectable_paths_dataset, injectable_lightcurve_collection.load_times_and_magnifications_from_path,
            injectable_lightcurve_collection.load_label_from_path)
        lightcurve_and_label = next(iter(lightcurve_and_label_dataset))
        assert lightcurve_and_label[0].numpy().shape == (3, 1)
        assert np.array_equal(lightcurve_and_label[0].numpy(), [[0.5], [3], [5.5]])  # Injected lightcurve 0
        assert np.array_equal(lightcurve_and_label[1].numpy(), [0])  # Injected label 0.

    def test_can_preprocess_injected_lightcurve(self, database):
        injectee_lightcurve_collection = database.training_injectee_lightcurve_collection
        injectable_lightcurve_collection = database.training_injectable_lightcurve_collections[0]
        # noinspection PyUnresolvedReferences
        injectee_lightcurve_path = injectee_lightcurve_collection.get_paths()[0]
        injectee_load_from_path_function = injectee_lightcurve_collection.load_times_and_fluxes_from_path
        # noinspection PyUnresolvedReferences
        injectable_lightcurve_path = injectable_lightcurve_collection.get_paths()[0]
        load_label_from_path_function = injectable_lightcurve_collection.load_label_from_path
        expected_label = load_label_from_path_function(Path())
        injectable_load_from_path_function = injectable_lightcurve_collection.load_times_and_magnifications_from_path
        lightcurve, label = database.preprocess_injected_lightcurve(
            injectee_load_from_path_function, injectable_load_from_path_function, load_label_from_path_function,
            tf.convert_to_tensor(str(injectee_lightcurve_path)), tf.convert_to_tensor(str(injectable_lightcurve_path)))
        assert lightcurve.shape == (3, 1)
        assert np.array_equal(lightcurve, [[0.5], [3], [5.5]])  # Injected lightcurve 0.
        assert np.array_equal(label, [expected_label])  # Injected label 0.

    def test_can_create_tensorflow_dataset_for_lightcurve_collection_paths(self, database):
        injectee_paths_dataset = database.generate_paths_dataset_from_lightcurve_collection(
            database.training_injectee_lightcurve_collection)
        assert next(iter(injectee_paths_dataset)).numpy() == b'injectee_path.ext'

    def test_lightcurve_collection_paths_dataset_is_repeated(self, database):
        with patch.object(database_module.tf.data.Dataset, 'repeat',
                          side_effect=lambda dataset: dataset, autospec=True) as mock_repeat:
            _ = database.generate_paths_dataset_from_lightcurve_collection(
                database.training_injectee_lightcurve_collection)
            assert mock_repeat.called

    def test_lightcurve_collection_paths_dataset_is_shuffled(self, database):
        with patch.object(database_module.tf.data.Dataset, 'shuffle',
                          side_effect=lambda dataset, buffer_size: dataset, autospec=True) as mock_shuffle:
            _ = database.generate_paths_dataset_from_lightcurve_collection(
                database.training_injectee_lightcurve_collection)
            assert mock_shuffle.called
            assert mock_shuffle.call_args[0][1] == database.shuffle_buffer_size

    def test_can_create_tensorflow_datasets_for_multiple_lightcurve_collections_paths(self, database):
        standard_paths_datasets = database.generate_paths_datasets_from_lightcurve_collection_list(
            database.training_standard_lightcurve_collections)
        assert next(iter(standard_paths_datasets[0])).numpy() == b'standard_path0.ext'
        assert next(iter(standard_paths_datasets[1])).numpy() == b'standard_path1.ext'

    def test_can_inject_signal_into_fluxes(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5])
        lightcurve_times = np.array([10, 20, 30, 40, 50])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                             signal_magnifications, signal_times)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5]))

    def test_inject_signal_errors_on_out_of_bounds(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        lightcurve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with pytest.raises(ValueError):
            database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                   signal_magnifications, signal_times)

    def test_inject_signal_can_be_told_to_allow_out_of_bounds(self, database):
        lightcurve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        lightcurve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        database.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            fluxes_with_injected_signal = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                                 signal_magnifications, signal_times)
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))

    def test_inject_signal_using_repeats_for_out_of_bounds(self, database):
        lightcurve_fluxes = np.array([1, 1, 1, 1, 1, 1, 1])
        lightcurve_times = np.array([10, 20, 30, 40, 50, 60, 70])
        signal_magnifications = np.array([1, 2])
        signal_times = np.array([0, 10])
        database.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0.6  # Make signal offset end up as 40
            fluxes_with_injected_signal0 = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                                  signal_magnifications, signal_times)
        assert np.array_equal(fluxes_with_injected_signal0, np.array([2, 1, 2, 1, 2, 1, 2]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0.8  # Make signal offset end up as 50
            fluxes_with_injected_signal1 = database.inject_signal_into_lightcurve(lightcurve_fluxes, lightcurve_times,
                                                                                  signal_magnifications, signal_times)
        assert np.array_equal(fluxes_with_injected_signal1, np.array([1, 2, 1, 2, 1, 2, 1]))

    def test_injected_signal_randomly_varies_injectable_portion_used_when_injectable_larger_than_injectee(self,
                                                                                                          database):
        injectee_fluxes = np.array([1, 2, 3])
        injectee_times = np.array([10, 20, 30])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            injected = database.inject_signal_into_lightcurve(injectee_fluxes, injectee_times,
                                                              injectable_magnifications, injectable_times)
            assert np.array_equal(injected, np.array([1, 4, 7]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 1
            injected = database.inject_signal_into_lightcurve(injectee_fluxes, injectee_times,
                                                              injectable_magnifications, injectable_times)
            assert np.array_equal(injected, np.array([5, 4, 3]))

    def test_injected_signal_randomly_varies_injection_location_when_injectee_larger_than_injectable(self, database):
        injectee_fluxes = np.array([1, 2, 3, 4, 5])
        injectee_times = np.array([10, 20, 30, 40, 50])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 10, 20])
        database.out_of_bounds_injection_handling = OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 0
            injected = database.inject_signal_into_lightcurve(injectee_fluxes, injectee_times,
                                                              injectable_magnifications, injectable_times)
            assert np.array_equal(injected, np.array([1, 8, 3, 4, 5]))
        with patch.object(database_module.np.random, 'random') as mock_random:
            mock_random.return_value = 1
            injected = database.inject_signal_into_lightcurve(injectee_fluxes, injectee_times,
                                                              injectable_magnifications, injectable_times)
            assert np.array_equal(injected, np.array([1, 2, 3, 10, 5]))

    def test_can_intersperse_datasets(self, database):
        dataset0 = tf.data.Dataset.from_tensor_slices([[0], [2], [4]])
        dataset1 = tf.data.Dataset.from_tensor_slices([[1], [3], [5]])
        interspersed_dataset = database.intersperse_datasets([dataset0, dataset1])
        assert list(interspersed_dataset) == [[0], [1], [2], [3], [4], [5]]

    def test_can_intersperse_zipped_example_label_datasets(self, database):
        examples_dataset0 = tf.data.Dataset.from_tensor_slices([[0, 0], [2, 2], [4, 4]])
        labels_dataset0 = tf.data.Dataset.from_tensor_slices([[0], [-2], [-4]])
        dataset0 = tf.data.Dataset.zip((examples_dataset0, labels_dataset0))
        examples_dataset1 = tf.data.Dataset.from_tensor_slices([[1, 1], [3, 3], [5, 5]])
        labels_dataset1 = tf.data.Dataset.from_tensor_slices([[-1], [-3], [-5]])
        dataset1 = tf.data.Dataset.zip((examples_dataset1, labels_dataset1))
        interspersed_dataset = database.intersperse_datasets([dataset0, dataset1])
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
    def test_database_can_generate_training_and_validation_datasets_with_only_standard_collections(self, database):
        database.training_injectee_lightcurve_collection = None
        database.training_injectable_lightcurve_collections = []
        database.validation_injectee_lightcurve_collection = None
        database.validation_injectable_lightcurve_collections = []
        training_dataset, validation_dataset = database.generate_datasets()
        training_batch = next(iter(training_dataset))
        training_batch_examples = training_batch[0]
        training_batch_labels = training_batch[1]
        assert training_batch_examples.shape == (database.batch_size, 3, 1)
        assert training_batch_labels.shape == (database.batch_size, 1)
        assert np.array_equal(training_batch_examples[0].numpy(), [[0], [1], [2]])  # Standard lightcurve 0.
        assert np.array_equal(training_batch_labels[0].numpy(), [0])  # Standard label 0.
        assert np.array_equal(training_batch_examples[1].numpy(), [[1], [2], [3]])  # Standard lightcurve 1.
        assert np.array_equal(training_batch_labels[1].numpy(), [1])  # Standard label 1.
        assert np.array_equal(training_batch_examples[2].numpy(), [[0], [1], [2]])  # Standard lightcurve 0.
        assert np.array_equal(training_batch_examples[3].numpy(), [[1], [2], [3]])  # Standard lightcurve 1.

    @pytest.mark.slow
    @pytest.mark.functional
    def test_can_generate_infer_path_and_lightcurve_dataset_from_paths_dataset_and_label(self, database):
        lightcurve_collection = database.training_standard_lightcurve_collections[0]
        paths_dataset = database.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
        path_and_lightcurve_dataset = database.generate_infer_path_and_lightcurve_dataset(
            paths_dataset, lightcurve_collection.load_times_and_fluxes_from_path)
        path_and_lightcurve = next(iter(path_and_lightcurve_dataset))
        assert np.array_equal(path_and_lightcurve[0].numpy(), b'standard_path0.ext')  # Standard path 0.
        assert path_and_lightcurve[1].numpy().shape == (3, 1)
        assert np.array_equal(path_and_lightcurve[1].numpy(), [[0], [1], [2]])  # Standard lightcurve 0.

    def test_can_preprocess_infer_lightcurve(self, database):
        lightcurve_collection = database.training_standard_lightcurve_collections[0]
        # noinspection PyUnresolvedReferences
        lightcurve_path = lightcurve_collection.get_paths()[0]
        expected_label = lightcurve_collection.label
        load_from_path_function = lightcurve_collection.load_times_and_fluxes_from_path
        path, lightcurve = database.preprocess_infer_lightcurve(load_from_path_function,
                                                                tf.convert_to_tensor(str(lightcurve_path)))
        assert np.array_equal(path, 'standard_path0.ext')  # Standard path 0.
        assert lightcurve.shape == (3, 1)
        assert np.array_equal(lightcurve, [[0], [1], [2]])  # Standard lightcurve 0.

    @pytest.mark.slow
    @pytest.mark.functional
    def test_generated_standard_and_infer_datasets_return_the_same_lightcurve(self, database):
        lightcurve_collection = database.training_standard_lightcurve_collections[0]
        paths_dataset0 = database.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
        label = lightcurve_collection.label
        lightcurve_and_label_dataset = database.generate_standard_lightcurve_and_label_dataset(
            paths_dataset0, lightcurve_collection.load_times_and_fluxes_from_path,
            lightcurve_collection.load_label_from_path)
        lightcurve_and_label = next(iter(lightcurve_and_label_dataset))
        paths_dataset1 = database.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
        path_and_lightcurve_dataset = database.generate_infer_path_and_lightcurve_dataset(
            paths_dataset1, lightcurve_collection.load_times_and_fluxes_from_path)
        path_and_lightcurve = next(iter(path_and_lightcurve_dataset))
        assert np.array_equal(lightcurve_and_label[0].numpy(), path_and_lightcurve[1].numpy())
