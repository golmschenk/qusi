import pytest

from ramjet.analysis.lisa_signal_to_noise_calculator import LisaSignalToNoiseCalculator


class TestLisaSignalToNoiseCalculator:
    def test_john_bakers_example_snr_is_computed_by_calculator(self):
        orbital_period = 10000 / 86400  # 10000 seconds to days.
        distance = 75  # Parsecs.
        mass0 = 0.25  # Solar masses.
        mass1 = 0.25  # Solar masses.
        calculator = LisaSignalToNoiseCalculator()
        median_snr = calculator.calculate_median_signal_to_noise_ratio(orbital_period, distance, mass0, mass1)
        assert pytest.approx(median_snr) == 1.0872541
