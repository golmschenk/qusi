import sys

import numpy as np

from qusi.internal.light_curve import LightCurve, remove_infinite_flux_data_points_from_light_curve, \
    make_light_curve_non_empty
from qusi.internal.light_curve_transforms import ensure_native_byteorder


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


def test_ensure_native_byteorder():
    native_byteorder = '>' if sys.byteorder == 'big' else '<'
    non_native_byteorder = '>' if native_byteorder == '<' else '<'
    non_native_byteorder_array = np.array([10., 20., 30.], dtype=f'{non_native_byteorder}f8')
    assert non_native_byteorder_array.dtype.byteorder not in ['|', '=', native_byteorder]
    ensured_array = ensure_native_byteorder(non_native_byteorder_array)
    assert ensured_array.dtype.byteorder in ['|', '=', native_byteorder]
    assert ensured_array[0] == 10
