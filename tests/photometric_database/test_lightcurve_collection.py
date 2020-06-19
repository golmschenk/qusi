from pathlib import Path

import numpy as np
import pytest

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


class TestLightcurveCollection:
    def test_has_necessary_attributes(self):
        lightcurve_collection = LightcurveCollection()
        assert hasattr(lightcurve_collection, 'get_paths')
        assert hasattr(lightcurve_collection, 'load_times_and_fluxes_from_path')
        assert hasattr(lightcurve_collection, 'load_times_and_magnifications_from_path')
        assert hasattr(lightcurve_collection, 'label')

    def test_calling_load_times_and_fluxes_from_path_without_setting_raises_error(self):
        lightcurve_collection = LightcurveCollection()
        with pytest.raises(NotImplementedError):
            _ = lightcurve_collection.load_times_and_fluxes_from_path(Path('path.ext'))
        lightcurve_collection.load_times_and_fluxes_from_path = lambda path: (np.array([]), np.array([]))
        try:
            _ = lightcurve_collection.load_times_and_fluxes_from_path(Path('path.ext'))
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')
            
    def test_load_times_and_fluxes_from_path_can_be_set_by_passing_to_init(self):
        lightcurve_collection = LightcurveCollection(
            function_to_load_times_and_fluxes_from_path=lambda path: (np.array([]), np.array([])))
        try:
            _ = lightcurve_collection.load_times_and_fluxes_from_path(Path('path.ext'))
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')
            
    def test_load_times_and_fluxes_from_path_can_be_set_by_subclassing(self):
        class SubclassLightcurveCollection(LightcurveCollection):
            def load_times_and_fluxes_from_path(self, path):
                return (np.array([]), np.array([]))
        lightcurve_collection = SubclassLightcurveCollection()
        try:
            _ = lightcurve_collection.load_times_and_fluxes_from_path(Path('path.ext'))
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_calling_load_times_and_magnifications_from_path_without_setting_raises_error(self):
        lightcurve_collection = LightcurveCollection()
        with pytest.raises(NotImplementedError):
            _ = lightcurve_collection.load_times_and_magnifications_from_path(Path('path.ext'))
        lightcurve_collection.load_times_and_magnifications_from_path = lambda path: (np.array([]), np.array([]))
        try:
            _ = lightcurve_collection.load_times_and_magnifications_from_path(Path('path.ext'))
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_load_times_and_magnifications_from_path_can_be_set_by_passing_to_init(self):
        lightcurve_collection = LightcurveCollection(
            function_to_load_times_and_magnifications_from_path=lambda path: (np.array([]), np.array([])))
        try:
            _ = lightcurve_collection.load_times_and_magnifications_from_path(Path('path.ext'))
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_load_times_and_magnifications_from_path_can_be_set_by_subclassing(self):
        class SubclassLightcurveCollection(LightcurveCollection):
            def load_times_and_magnifications_from_path(self, path):
                return (np.array([]), np.array([]))

        lightcurve_collection = SubclassLightcurveCollection()
        try:
            _ = lightcurve_collection.load_times_and_magnifications_from_path(Path('path.ext'))
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_calling_get_paths_without_setting_raises_error(self):
        lightcurve_collection = LightcurveCollection()
        with pytest.raises(NotImplementedError):
            _ = lightcurve_collection.get_paths()
        lightcurve_collection.get_paths = lambda: []
        try:
            _ = lightcurve_collection.get_paths()
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_get_paths_can_be_set_by_passing_to_init(self):
        lightcurve_collection = LightcurveCollection(function_to_get_paths=lambda: [])
        try:
            _ = lightcurve_collection.get_paths()
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_get_paths_can_be_set_by_subclassing(self):
        class SubclassLightcurveCollection(LightcurveCollection):
            def get_paths(self):
                return []
        lightcurve_collection = SubclassLightcurveCollection()
        try:
            _ = lightcurve_collection.get_paths()
        except NotImplementedError:
            pytest.fail('`NotImplementedError` raised when it should be considered implemented.')

    def test_generating_a_synthetic_signal_from_a_real_signal_does_not_invert_negative_lightcurve_shapes(self):
        lightcurve_collection = LightcurveCollection()
        times = np.arange(5, dtype=np.float32)
        unnormalized_positive_lightcurve_fluxes = np.array([10, 20, 30, 25, 15], dtype=np.float32)
        normalized_positive_lightcurve_fluxes, _ = lightcurve_collection.generate_synthetic_signal_from_real_data(
            unnormalized_positive_lightcurve_fluxes, times)
        assert normalized_positive_lightcurve_fluxes.argmax() == 2
        assert normalized_positive_lightcurve_fluxes.argmin() == 0
        unnormalized_negative_lightcurve_fluxes = np.array([-30, -20, -10, -15, -25], dtype=np.float32)
        normalized_negative_lightcurve_fluxes, _ = lightcurve_collection.generate_synthetic_signal_from_real_data(
            unnormalized_negative_lightcurve_fluxes, times)
        assert normalized_negative_lightcurve_fluxes.argmax() == 2
        assert normalized_negative_lightcurve_fluxes.argmin() == 0