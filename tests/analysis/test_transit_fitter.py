
from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd

import ramjet.analysis.transit_fitter
from ramjet.analysis.transit_fitter import TransitFitter


class TestTransitFitter:
    @pytest.fixture
    @patch.object(ramjet.analysis.transit_fitter, 'TessDataInterface')
    def transit_fitter(self, mock_tess_data_interface_class) -> TransitFitter:
        """
        Sets up the transit fitter for use in a test.

        :return: The transit fitter.
        """
        mock_tess_data_interface = Mock()
        mock_tess_data_interface.download_lightcurve = Mock()
        mock_tess_data_interface.load_fluxes_flux_errors_and_times_from_fits_file = Mock(return_value=(
            np.array([1, 2, 3, 4, 5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([10, 20, 30, 40, 50])
        ))
        mock_tess_data_interface.get_tess_input_catalog_row = Mock(return_value=pd.Series({'rad': 1}))
        mock_tess_data_interface_class.return_value = mock_tess_data_interface
        return TransitFitter(tic_id=23324827, sector=9)

    def test_can_calculate_period_from_approximate_event_times(self, transit_fitter):
        event_times = [3, 4.9, 7.1, 11, 17.1, 19.1]
        epoch, period = transit_fitter.calculate_epoch_and_period_from_approximate_event_times(event_times)
        assert epoch == 3
        assert period == pytest.approx(2)
