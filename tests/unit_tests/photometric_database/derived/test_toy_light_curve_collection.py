import pytest

from ramjet.photometric_database.derived.toy_light_curve_collection import ToyLightCurve


class TestToyLightCurveCollection:
    def test_can_create_flat_light_curve(self):
        light_curve = ToyLightCurve.flat()
        assert light_curve.fluxes[10] == 1
        assert light_curve.times.shape[0] == 100

    def test_can_create_sine_wave_light_curve(self):
        light_curve = ToyLightCurve.sine_wave()
        assert light_curve.fluxes[0] == pytest.approx(1)
        assert light_curve.fluxes[25] == pytest.approx(1.5)
        assert light_curve.fluxes[50] == pytest.approx(1)
        assert light_curve.times.shape[0] == 100
