import numpy as np

from qusi.internal.light_curve import LightCurve, remove_infinite_flux_data_points_from_light_curve, \
    make_light_curve_non_empty


def test_remove_infinite_flux_data_points_from_light_curve():
    times = np.array([0.0, 1.0, 2.0])
    fluxes = np.array([0.0, np.inf, 20.0])
    light_curve = LightCurve.new(
        times=times,
        fluxes=fluxes,
    )
    updated_light_curve = remove_infinite_flux_data_points_from_light_curve(light_curve=light_curve)
    expected_times = np.array([0.0, 2.0])
    expected_fluxes = np.array([0.0, 20.0])
    assert np.array_equal(updated_light_curve.times, expected_times)
    assert np.array_equal(updated_light_curve.fluxes, expected_fluxes)


def test_make_light_curve_non_empty():
    times = np.array([], dtype=np.float32)
    fluxes = np.array([], dtype=np.float32)
    light_curve = LightCurve.new(
        times=times,
        fluxes=fluxes,
    )
    updated_light_curve = make_light_curve_non_empty(light_curve)
    assert updated_light_curve.fluxes.shape[0] > 0
