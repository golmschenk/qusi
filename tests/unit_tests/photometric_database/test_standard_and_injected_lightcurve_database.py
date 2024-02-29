from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import ramjet.photometric_database.standard_and_injected_light_curve_database as database_module
from ramjet.photometric_database.light_curve_collection import LightCurveCollection
from ramjet.photometric_database.light_curve_dataset_manipulations import (
    OutOfBoundsInjectionHandlingMethod,
)
from ramjet.photometric_database.standard_and_injected_light_curve_database import (
    StandardAndInjectedLightCurveDatabase,
)


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
        standard_light_curve_collection0.get_paths = lambda: [
            Path("standard_path0.ext")
        ]
        standard_light_curve_collection0.load_times_and_fluxes_from_path = (
            lambda _path: (
                np.array([10, 20, 30]),
                np.array([0, 1, 2]),
            )
        )
        standard_light_curve_collection0.label = 0
        standard_light_curve_collection1 = LightCurveCollection()
        standard_light_curve_collection1.get_paths = lambda: [
            Path("standard_path1.ext")
        ]
        standard_light_curve_collection1.load_times_and_fluxes_from_path = (
            lambda _path: (
                np.array([20, 30, 40]),
                np.array([1, 2, 3]),
            )
        )
        standard_light_curve_collection1.label = 1
        injectee_light_curve_collection = LightCurveCollection()
        injectee_light_curve_collection.get_paths = lambda: [Path("injectee_path.ext")]
        injectee_light_curve_collection.load_times_and_fluxes_from_path = (
            lambda _path: (
                np.array([30, 40, 50]),
                np.array([2, 3, 4]),
            )
        )
        injectee_light_curve_collection.label = 0
        injectable_light_curve_collection0 = LightCurveCollection()
        injectable_light_curve_collection0.get_paths = lambda: [
            Path("injectable_path0.ext")
        ]
        injectable_light_curve_collection0.load_times_and_magnifications_from_path = (
            lambda _path: (
                np.array([0, 10, 20]),
                np.array([0.5, 1, 1.5]),
            )
        )
        injectable_light_curve_collection0.label = 0
        injectable_light_curve_collection1 = LightCurveCollection()
        injectable_light_curve_collection1.get_paths = lambda: [
            Path("injectable_path1.ext")
        ]
        injectable_light_curve_collection1.load_times_and_magnifications_from_path = (
            lambda _path: (
                np.array([0, 10, 20, 30]),
                np.array([0, 1, 1, 0]),
            )
        )
        injectable_light_curve_collection1.label = 1
        database.training_standard_light_curve_collections = [
            standard_light_curve_collection0,
            standard_light_curve_collection1,
        ]
        database.training_injectee_light_curve_collection = (
            injectee_light_curve_collection
        )
        database.training_injectable_light_curve_collections = [
            injectable_light_curve_collection0,
            injectable_light_curve_collection1,
        ]
        database.validation_standard_light_curve_collections = [
            standard_light_curve_collection1
        ]
        database.validation_injectee_light_curve_collection = (
            injectee_light_curve_collection
        )
        database.validation_injectable_light_curve_collections = [
            injectable_light_curve_collection1
        ]
        # Setup simplified database settings
        database.batch_size = 4
        database.time_steps_per_example = 3
        database.number_of_parallel_processes_per_map = 1

        def mock_window(dataset, batch_size):
            return dataset.batch(batch_size)

        database.window_dataset_for_zipped_example_and_label_dataset = (
            mock_window  # Disable windowing.
        )
        database.normalize_on_percentiles = (
            lambda fluxes: fluxes
        )  # Don't normalize values to keep it simple.
        return database

    @pytest.fixture
    def deterministic_database(
        self, database_with_collections
    ) -> StandardAndInjectedLightCurveDatabase:
        """A fixture of a deterministic database with light_curve collections pre-prepared."""
        database_with_collections.remove_random_elements = lambda x: x
        database_with_collections.randomly_roll_elements = lambda x: x
        return database_with_collections

    def test_database_has_light_curve_collection_properties(self):
        database = StandardAndInjectedLightCurveDatabase()
        assert hasattr(database, "training_standard_light_curve_collections")
        assert hasattr(database, "training_injectee_light_curve_collection")
        assert hasattr(database, "training_injectable_light_curve_collections")
        assert hasattr(database, "validation_standard_light_curve_collections")
        assert hasattr(database, "validation_injectee_light_curve_collection")
        assert hasattr(database, "validation_injectable_light_curve_collections")

    def test_can_inject_signal_into_fluxes(self):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5])
        light_curve_times = np.array([10, 20, 30, 40, 50])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = database_module.inject_signal_into_light_curve(
            light_curve_times, light_curve_fluxes, signal_times, signal_magnifications
        )
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5]))

    def test_inject_signal_errors_on_out_of_bounds(self):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with pytest.raises(
            ValueError,
            match=r"A value \(-?[0-9]\d*(\.\d+)?\) in x_new is below the interpolation "
            "range's minimum value \\(-?[0-9]\\d*(\\.\\d+)?\\)",
        ):
            database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
            )

    def test_inject_signal_can_be_told_to_allow_out_of_bounds(self):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0
            fluxes_with_injected_signal = database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION,
            )
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))

    def test_inject_signal_using_repeats_for_out_of_bounds(self):
        light_curve_fluxes = np.array([1, 1, 1, 1, 1, 1, 1])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60, 70])
        signal_magnifications = np.array([1, 2])
        signal_times = np.array([0, 10])
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0.6  # Make signal offset end up as 40
            fluxes_with_injected_signal0 = database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL,
            )
        assert np.array_equal(
            fluxes_with_injected_signal0, np.array([2, 1, 2, 1, 2, 1, 2])
        )
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0.8  # Make signal offset end up as 50
            fluxes_with_injected_signal1 = database_module.inject_signal_into_light_curve(
                light_curve_times,
                light_curve_fluxes,
                signal_times,
                signal_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL,
            )
        assert np.array_equal(
            fluxes_with_injected_signal1, np.array([1, 2, 1, 2, 1, 2, 1])
        )

    def test_injected_signal_randomly_varies_injectable_portion_used_when_injectable_larger_than_injectee(
        self,
    ):
        injectee_fluxes = np.array([1, 2, 3])
        injectee_times = np.array([10, 20, 30])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications,
            )
            assert np.array_equal(injected, np.array([1, 4, 7]))
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 1
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications,
            )
            assert np.array_equal(injected, np.array([5, 4, 3]))

    def test_injected_signal_randomly_varies_injection_location_when_injectee_larger_than_injectable(
        self,
    ):
        injectee_fluxes = np.array([1, 2, 3, 4, 5])
        injectee_times = np.array([10, 20, 30, 40, 50])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 10, 20])
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION,
            )
            assert np.array_equal(injected, np.array([1, 8, 3, 4, 5]))
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 1
            injected = database_module.inject_signal_into_light_curve(
                injectee_times,
                injectee_fluxes,
                injectable_times,
                injectable_magnifications,
                out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION,
            )
            assert np.array_equal(injected, np.array([1, 2, 3, 10, 5]))

    def test_database_can_inject_signal_into_fluxes(self, database_with_collections):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5])
        light_curve_times = np.array([10, 20, 30, 40, 50])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        fluxes_with_injected_signal = (
            database_with_collections.inject_signal_into_light_curve(
                light_curve_fluxes,
                light_curve_times,
                signal_magnifications,
                signal_times,
            )
        )
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5]))

    def test_database_inject_signal_errors_on_out_of_bounds(
        self, database_with_collections
    ):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        with pytest.raises(
            ValueError,
            match=r"A value \(-?[0-9]\d*(\.\d+)?\) in x_new is below the interpolation "
            "range's minimum value \\(-?[0-9]\\d*(\\.\\d+)?\\)",
        ):
            database_with_collections.inject_signal_into_light_curve(
                light_curve_fluxes,
                light_curve_times,
                signal_magnifications,
                signal_times,
            )

    def test_database_inject_signal_can_be_told_to_allow_out_of_bounds(
        self, database_with_collections
    ):
        light_curve_fluxes = np.array([1, 2, 3, 4, 5, 3])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60])
        signal_magnifications = np.array([1, 3, 1])
        signal_times = np.array([0, 20, 40])
        database_with_collections.out_of_bounds_injection_handling = (
            OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        )
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0
            fluxes_with_injected_signal = (
                database_with_collections.inject_signal_into_light_curve(
                    light_curve_fluxes,
                    light_curve_times,
                    signal_magnifications,
                    signal_times,
                )
            )
        assert np.array_equal(fluxes_with_injected_signal, np.array([1, 5, 9, 7, 5, 3]))

    def test_database_inject_signal_using_repeats_for_out_of_bounds(
        self, database_with_collections
    ):
        light_curve_fluxes = np.array([1, 1, 1, 1, 1, 1, 1])
        light_curve_times = np.array([10, 20, 30, 40, 50, 60, 70])
        signal_magnifications = np.array([1, 2])
        signal_times = np.array([0, 10])
        database_with_collections.out_of_bounds_injection_handling = (
            OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
        )
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0.6  # Make signal offset end up as 40
            fluxes_with_injected_signal0 = (
                database_with_collections.inject_signal_into_light_curve(
                    light_curve_fluxes,
                    light_curve_times,
                    signal_magnifications,
                    signal_times,
                )
            )
        assert np.array_equal(
            fluxes_with_injected_signal0, np.array([2, 1, 2, 1, 2, 1, 2])
        )
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0.8  # Make signal offset end up as 50
            fluxes_with_injected_signal1 = (
                database_with_collections.inject_signal_into_light_curve(
                    light_curve_fluxes,
                    light_curve_times,
                    signal_magnifications,
                    signal_times,
                )
            )
        assert np.array_equal(
            fluxes_with_injected_signal1, np.array([1, 2, 1, 2, 1, 2, 1])
        )

    def test_database_injected_signal_randomly_varies_injectable_portion_used_when_injectable_larger_than_injectee(
        self, database_with_collections
    ):
        injectee_fluxes = np.array([1, 2, 3])
        injectee_times = np.array([10, 20, 30])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 20, 40])
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0
            injected = database_with_collections.inject_signal_into_light_curve(
                injectee_fluxes,
                injectee_times,
                injectable_magnifications,
                injectable_times,
            )
            assert np.array_equal(injected, np.array([1, 4, 7]))
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 1
            injected = database_with_collections.inject_signal_into_light_curve(
                injectee_fluxes,
                injectee_times,
                injectable_magnifications,
                injectable_times,
            )
            assert np.array_equal(injected, np.array([5, 4, 3]))

    def test_database_injected_signal_randomly_varies_injection_location_when_injectee_larger_than_injectable(
        self, database_with_collections
    ):
        injectee_fluxes = np.array([1, 2, 3, 4, 5])
        injectee_times = np.array([10, 20, 30, 40, 50])
        injectable_magnifications = np.array([1, 3, 1])
        injectable_times = np.array([0, 10, 20])
        database_with_collections.out_of_bounds_injection_handling = (
            OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
        )
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 0
            injected = database_with_collections.inject_signal_into_light_curve(
                injectee_fluxes,
                injectee_times,
                injectable_magnifications,
                injectable_times,
            )
            assert np.array_equal(injected, np.array([1, 8, 3, 4, 5]))
        with patch.object(database_module.np.random, "random") as mock_random:
            mock_random.return_value = 1
            injected = database_with_collections.inject_signal_into_light_curve(
                injectee_fluxes,
                injectee_times,
                injectable_magnifications,
                injectable_times,
            )
            assert np.array_equal(injected, np.array([1, 2, 3, 10, 5]))

    @pytest.mark.parametrize(
        ("original_label", "expected_label"),
        [
            (0, np.array([0])),
            ([0], np.array([0])),
            (np.array([0]), np.array([0])),
            ([0, 0], np.array([0, 0])),
        ],
    )
    def test_expand_label_to_training_dimensions(self, original_label, expected_label):
        label = database_module.expand_label_to_training_dimensions(original_label)
        assert type(label) is np.ndarray
        assert np.array_equal(label, expected_label)
