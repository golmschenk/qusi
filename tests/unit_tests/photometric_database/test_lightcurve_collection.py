from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from ramjet.photometric_database.light_curve_collection import (
    LightCurveCollection,
    LightCurveCollectionMethodNotImplementedError,
)


class TestLightCurveCollection:
    def test_has_necessary_attributes(self):
        light_curve_collection = LightCurveCollection()
        assert hasattr(light_curve_collection, "get_paths")
        assert hasattr(light_curve_collection, "load_times_and_fluxes_from_path")
        assert hasattr(
            light_curve_collection, "load_times_and_magnifications_from_path"
        )
        assert hasattr(light_curve_collection, "label")

    def test_calling_load_times_and_fluxes_from_path_without_setting_raises_error(self):
        light_curve_collection = LightCurveCollection()
        with pytest.raises(LightCurveCollectionMethodNotImplementedError):
            _ = light_curve_collection.load_times_and_fluxes_from_path(Path("path.ext"))
        light_curve_collection.load_times_and_fluxes_from_path = lambda _path: (
            np.array([]),
            np.array([]),
        )
        try:
            _ = light_curve_collection.load_times_and_fluxes_from_path(Path("path.ext"))
        except LightCurveCollectionMethodNotImplementedError:
            pytest.fail(
                "`LightCurveCollectionMethodNotImplementedError` raised when it should not be."
            )

    def test_load_times_and_fluxes_from_path_can_be_set_by_subclassing(self):
        class SubclassLightCurveCollection(LightCurveCollection):
            def load_times_and_fluxes_from_path(self, _path):
                return (np.array([]), np.array([]))

        light_curve_collection = SubclassLightCurveCollection()
        try:
            _ = light_curve_collection.load_times_and_fluxes_from_path(Path("path.ext"))
        except LightCurveCollectionMethodNotImplementedError:
            pytest.fail(
                "`LightCurveCollectionMethodNotImplementedError` raised when it should not be."
            )

    def test_calling_load_times_and_magnifications_from_path_without_setting_raises_error(
        self,
    ):
        light_curve_collection = LightCurveCollection()
        with pytest.raises(LightCurveCollectionMethodNotImplementedError):
            _ = light_curve_collection.load_times_and_magnifications_from_path(
                Path("path.ext")
            )
        light_curve_collection.load_times_and_magnifications_from_path = lambda _path: (
            np.array([]),
            np.array([]),
        )
        try:
            _ = light_curve_collection.load_times_and_magnifications_from_path(
                Path("path.ext")
            )
        except LightCurveCollectionMethodNotImplementedError:
            pytest.fail(
                "`LightCurveCollectionMethodNotImplementedError` raised when it should not be."
            )

    def test_load_times_and_magnifications_from_path_can_be_set_by_subclassing(self):
        class SubclassLightCurveCollection(LightCurveCollection):
            def load_times_and_magnifications_from_path(self, _path):
                return (np.array([]), np.array([]))

        light_curve_collection = SubclassLightCurveCollection()
        try:
            _ = light_curve_collection.load_times_and_magnifications_from_path(
                Path("path.ext")
            )
        except LightCurveCollectionMethodNotImplementedError:
            pytest.fail(
                "`LightCurveCollectionMethodNotImplementedError` raised when it should not be."
            )

    def test_calling_get_paths_without_setting_returns_the_paths_attribute(self):
        light_curve_collection = LightCurveCollection()
        stub_paths = Mock()
        light_curve_collection.paths = stub_paths
        paths0 = light_curve_collection.get_paths()
        assert paths0 is stub_paths
        light_curve_collection.get_paths = list
        paths1 = light_curve_collection.get_paths()
        assert paths1 == []

    def test_get_paths_can_be_set_by_subclassing(self):
        class SubclassLightCurveCollection(LightCurveCollection):
            def get_paths(self):
                return []

        light_curve_collection = SubclassLightCurveCollection()
        try:
            _ = light_curve_collection.get_paths()
        except LightCurveCollectionMethodNotImplementedError:
            pytest.fail(
                "`LightCurveCollectionMethodNotImplementedError` raised when it should not be."
            )

    def test_generating_a_synthetic_signal_from_a_real_signal_does_not_invert_negative_light_curve_shapes(
        self,
    ):
        light_curve_collection = LightCurveCollection()
        times = np.arange(5, dtype=np.float32)
        unnormalized_positive_light_curve_fluxes = np.array(
            [10, 20, 30, 25, 15], dtype=np.float32
        )
        (
            normalized_positive_light_curve_fluxes,
            _,
        ) = light_curve_collection.generate_synthetic_signal_from_real_data(
            unnormalized_positive_light_curve_fluxes, times
        )
        assert normalized_positive_light_curve_fluxes.argmax() == 2
        assert normalized_positive_light_curve_fluxes.argmin() == 0
        unnormalized_negative_light_curve_fluxes = np.array(
            [-30, -20, -10, -15, -25], dtype=np.float32
        )
        (
            normalized_negative_light_curve_fluxes,
            _,
        ) = light_curve_collection.generate_synthetic_signal_from_real_data(
            unnormalized_negative_light_curve_fluxes, times
        )
        assert normalized_negative_light_curve_fluxes.argmax() == 2
        assert normalized_negative_light_curve_fluxes.argmin() == 0

    def test_load_times_fluxes_and_flux_errors_defaults_to_just_the_times_and_fluxes_loading(
        self,
    ):
        light_curve_collection = LightCurveCollection()
        light_curve_collection.load_times_and_fluxes_from_path = lambda _path: (
            [0],
            [1],
        )
        (
            times,
            fluxes,
            flux_errors,
        ) = light_curve_collection.load_times_fluxes_and_flux_errors_from_path(
            Path("fake")
        )
        assert np.array_equal(times, [0])
        assert np.array_equal(fluxes, [1])
        assert flux_errors is None

    def test_load_times_magnifications_and_magnification_errors_defaults_to_just_the_times_and_fluxes_loading(
        self,
    ):
        light_curve_collection = LightCurveCollection()
        light_curve_collection.load_times_and_magnifications_from_path = lambda _path: (
            [0],
            [1],
        )
        (
            times,
            magnifications,
            magnification_errors,
        ) = light_curve_collection.load_times_magnifications_and_magnification_errors_from_path(
            Path("fake")
        )
        assert np.array_equal(times, [0])
        assert np.array_equal(magnifications, [1])
        assert magnification_errors is None
