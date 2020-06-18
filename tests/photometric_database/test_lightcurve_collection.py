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

