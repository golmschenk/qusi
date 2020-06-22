from unittest.mock import patch, Mock, MagicMock

import ramjet.photometric_database.derived.self_lensing_binary_synthetic_signals_lightcurve_collection as module
from ramjet.photometric_database.derived.self_lensing_binary_synthetic_signals_lightcurve_collection import \
    SelfLensingBinarySyntheticSignalsLightcurveCollection


class TestSelfLensingBinarySyntheticSignalsLightcurveCollection:
    def test_can_request_download_of_synthetic_csv_data_from_http(self):
        with patch.object(module.urllib.request, 'urlretrieve') as mock_urlretrieve:
            with patch.object(module.tarfile, 'open') as mock_open:
                mock_extractall = Mock()
                # Mock the tarfile open context manager.
                mock_open.return_value.__enter__.return_value.extractall = mock_extractall
                lightcurve_collection = SelfLensingBinarySyntheticSignalsLightcurveCollection()
                lightcurve_collection.download_csv_files()
                expected_save_path = lightcurve_collection.data_directory.joinpath('synthetic_signals_csv_files.tar')
                assert mock_urlretrieve.called
                assert mock_urlretrieve.call_args[0][1] == str(expected_save_path)
                assert mock_open.called
                assert mock_open.call_args[0][0] == expected_save_path
                assert mock_extractall.called
                assert mock_extractall.call_args[0][0] == lightcurve_collection.data_directory
