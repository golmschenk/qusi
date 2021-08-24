from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd

import ramjet.analysis.transit_fitter
from ramjet.analysis.transit_fitter import TransitFitter


class TestTransitFitter:
    @pytest.fixture
    @patch.object(ramjet.analysis.transit_fitter, 'TessDataInterface')
    @patch.object(ramjet.analysis.transit_fitter, 'TessToiDataInterface')
    def transit_fitter(self, mock_tess_toi_data_interface_class, mock_tess_data_interface_class) -> TransitFitter:
        """
        Sets up the transit fitter for use in a test.

        :return: The transit fitter.
        """
        mock_tess_data_interface = Mock()
        mock_tess_data_interface.download_light_curve = Mock()
        mock_tess_data_interface.load_fluxes_flux_errors_and_times_from_fits_file = Mock(return_value=(
            np.array([1, 2, 3, 4, 5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([10, 20, 30, 40, 50])
        ))
        mock_tess_toi_data_interface_class.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id = Mock(
            return_value=pd.DataFrame())
        mock_tess_data_interface.get_tess_input_catalog_row = Mock(return_value=pd.Series({'rad': 1}))
        mock_tess_data_interface.get_sectors_target_appears_in = Mock(return_value=[9])
        mock_tess_data_interface_class.return_value = mock_tess_data_interface
        return TransitFitter(tic_id=23324827, sectors=[9])

    def test_can_calculate_period_from_approximate_event_times(self, transit_fitter):
        event_times = [3, 4.9, 7.1, 11, 17.1, 19.01]
        epoch, period = transit_fitter.calculate_epoch_and_period_from_approximate_event_times(event_times)
        assert epoch == 3
        assert period == pytest.approx(2, rel=0.1)

    def test_can_fold_times_based_on_epoch_and_period(self, transit_fitter):
        times = np.array([1, 2, 3, 4, 5, 6])
        folded_times = transit_fitter.fold_times(times=times, epoch=2, period=3)
        assert np.array_equal(folded_times, [-1, 0, 1, -1, 0, 1])

    def test_can_round_series_to_significant_digits(self, transit_fitter):
        series = pd.Series([1.2345, 0.00012345, 12345000])
        rounded_series_3_significant_figures = transit_fitter.round_series_to_significant_figures(series,
                                                                                                  significant_figures=3)
        assert rounded_series_3_significant_figures.equals(pd.Series([1.23, 0.000123, 12300000]))
        rounded_series_4_significant_figures = transit_fitter.round_series_to_significant_figures(series,
                                                                                                  significant_figures=4)
        assert rounded_series_4_significant_figures.equals(pd.Series([1.234, 0.0001234, 12340000]))
