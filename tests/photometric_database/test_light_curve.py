import pytest

from ramjet.photometric_database.light_curve import LightCurve


class TestLightCurve:
    def test_fluxes_returns_the_only_fluxes_entry_from_the_dictionary_if_there_is_only_one(self):
        light_curve = LightCurve()
        light_curve.fluxes_dictionary = {'a': [0, 1]}
        assert light_curve.fluxes == [0, 1]

    def test_fluxes_returns_the_default_flux_type_when_available(self):
        light_curve = LightCurve()
        light_curve.fluxes_dictionary = {'a': [0, 1], 'b': [2, 3]}
        light_curve.default_flux_type = 'b'
        assert light_curve.fluxes == [2, 3]

    def test_fluxes_errors_if_no_fluxes_are_available(self):
        light_curve = LightCurve()
        with pytest.raises(ValueError):
            _ = light_curve.fluxes

    def test_fluxes_errors_if_multiple_flux_types_are_available_but_none_are_default(self):
        light_curve = LightCurve()
        light_curve.fluxes_dictionary = {'a': [0, 1], 'b': [2, 3]}
        with pytest.raises(ValueError):
            _ = light_curve.fluxes
