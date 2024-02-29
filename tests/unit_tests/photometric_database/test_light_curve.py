import numpy as np
import pandas as pd
import pytest

from ramjet.photometric_database.light_curve import LightCurve


class TestLightCurve:
    def test_times_are_drawn_from_light_curve_data_frame_when_column_name_is_set(self):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame({"time_column": [0, 1]})
        light_curve.time_column_name = "time_column"
        assert np.array_equal(light_curve.times, [0, 1])

    def test_fluxes_are_drawn_from_light_curve_data_frame_when_column_names_is_set(
        self,
    ):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame({"flux_column": [0, 1]})
        light_curve.flux_column_names = ["flux_column"]
        assert np.array_equal(light_curve.fluxes, [0, 1])

    def test_times_error_when_column_name_is_not_set(self):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame({"time_column": [0, 1]})
        with pytest.raises(KeyError):
            _ = light_curve.times

    def test_fluxes_error_when_column_name_is_not_set(self):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame({"flux_column": [0, 1]})
        with pytest.raises(IndexError):
            _ = light_curve.fluxes

    def test_time_column_name_is_set_when_times_are_manually_set(self):
        light_curve = LightCurve()
        assert light_curve.time_column_name is None
        light_curve.times = [0, 1]
        assert light_curve.time_column_name is not None

    def test_flux_column_name_is_set_when_times_are_manually_set(self):
        light_curve = LightCurve()
        assert len(light_curve.flux_column_names) == 0
        light_curve.fluxes = [0, 1]
        assert len(light_curve.flux_column_names) == 1

    def test_setting_times_sets_existing_named_time_column_if_one_exists(self):
        light_curve = LightCurve()
        light_curve.time_column_name = "time_column"
        light_curve.times = [0, 1]
        assert np.array_equal(light_curve.data_frame["time_column"].values, [0, 1])

    def test_setting_fluxes_sets_existing_named_flux_column_if_one_exists(self):
        light_curve = LightCurve()
        light_curve.flux_column_names = ["flux_column"]
        light_curve.fluxes = [0, 1]
        assert np.array_equal(light_curve.data_frame["flux_column"].values, [0, 1])

    def test_can_convert_column_to_relative_scale(self):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame({"a": [1, 2, 3]})
        light_curve.convert_column_to_relative_scale("a")
        assert np.array_equal(light_curve.data_frame["a"].values, [0.5, 1, 1.5])

    def test_can_convert_columns_to_relative_scale(self):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame(
            {"a": [1, 2, 3], "b": [-1, -2, -3], "c": [1, 2, 3]}
        )
        light_curve.convert_columns_to_relative_scale(["a", "b"])
        assert np.array_equal(light_curve.data_frame["a"].values, [0.5, 1, 1.5])
        assert np.array_equal(light_curve.data_frame["b"].values, [0.5, 1, 1.5])
        assert np.array_equal(light_curve.data_frame["c"].values, [1, 2, 3])

    def test_can_convert_column_to_relative_scale_with_nans(self):
        light_curve = LightCurve()
        light_curve.data_frame = pd.DataFrame({"a": [1, 2, 3, np.nan]})
        light_curve.convert_column_to_relative_scale("a")
        light_curve_values = light_curve.data_frame["a"].values
        expected_values = np.array([0.5, 1, 1.5, np.nan])
        assert np.allclose(light_curve_values, expected_values, equal_nan=True)
