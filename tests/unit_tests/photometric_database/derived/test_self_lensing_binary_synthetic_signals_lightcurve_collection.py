from unittest.mock import Mock, patch

import numpy as np

import ramjet.photometric_database.derived.self_lensing_binary_synthetic_signals_light_curve_collection as module
from ramjet.photometric_database.derived.self_lensing_binary_synthetic_signals_light_curve_collection import (
    SelfLensingBinarySyntheticSignalsLightCurveCollection,
)


class TestSelfLensingBinarySyntheticSignalsLightCurveCollection:
    @patch.object(module, "Path")
    def test_can_request_download_of_synthetic_csv_data_from_http(self, mock_path):
        with (
            patch.object(module.urllib.request, "urlretrieve") as mock_urlretrieve,
            patch.object(module.tarfile, "open") as mock_open,
        ):
            mock_path.open = Mock()
            mock_path.glob = Mock(return_value=[])
            mock_extractall = Mock()
            # Mock the tarfile open context manager.
            mock_open.return_value.__enter__.return_value.extractall = mock_extractall
            light_curve_collection = (
                SelfLensingBinarySyntheticSignalsLightCurveCollection()
            )
            light_curve_collection.download_csv_files()
            expected_save_path = light_curve_collection.data_directory.joinpath(
                "synthetic_signals_csv_files.tar"
            )
            assert mock_urlretrieve.called
            assert mock_urlretrieve.call_args[0][1] == str(expected_save_path)
            assert mock_open.called
            assert mock_open.call_args[0][0] == expected_save_path
            assert mock_extractall.called
            assert (
                mock_extractall.call_args[0][0] == light_curve_collection.data_directory
            )


class TestReversedSelfLensingBinarySyntheticSignalsLightCurveCollection:
    def test_reversal_of_signals(self):
        times = np.array([0, 10, 20, 30, 40, 50])
        magnitudes = np.array([0, 1, 2, 3, 4, 5])
        light_curve_collection = (
            module.ReversedSelfLensingBinarySyntheticSignalsLightCurveCollection()
        )
        reversed_times, reversed_magnitudes = light_curve_collection.reverse_signal(
            times, magnitudes
        )
        assert np.array_equal(reversed_times, [0, 10, 20, 30, 40, 50])
        assert np.array_equal(reversed_magnitudes, [5, 4, 3, 2, 1, 0])
