from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

import ramjet.data_interface.tess_toi_data_interface as module
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface


class TestTessToiDataInterface:
    @pytest.fixture
    def data_interface(self) -> TessToiDataInterface:
        """
        A fixture of the data interface under test.

        :return: The data interface.
        """
        return TessToiDataInterface()

    @pytest.mark.slow
    @pytest.mark.external
    def test_can_retrieve_the_tess_toi_dispositions(self, data_interface):
        dispositions = data_interface.toi_dispositions
        target_dispositions = dispositions[dispositions["TIC ID"] == 307210830]
        assert target_dispositions["Disposition"].iloc[0] == "CP"
        target_planet_sectors = target_dispositions["Sector"].unique()
        assert 1 not in target_planet_sectors
        assert 2 in target_planet_sectors
        assert 3 not in target_planet_sectors
        assert 5 in target_planet_sectors
        assert 31 not in target_planet_sectors
        assert 32 in target_planet_sectors

    @pytest.mark.slow
    @pytest.mark.external
    def test_can_get_exofop_planet_disposition_for_tic_id(self, data_interface):
        mock_dispositions = pd.DataFrame(
            {"TIC ID": [231663901, 266980320], "Disposition": ["KP", "CP"]}
        )
        data_interface.retrieve_toi_dispositions_from_exofop = Mock(
            return_value=mock_dispositions
        )
        dispositions0 = (
            data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(
                tic_id=231663901
            )
        )
        assert dispositions0["Disposition"].iloc[0] == "KP"
        dispositions1 = (
            data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(
                tic_id=266980320
            )
        )
        assert dispositions1["Disposition"].iloc[0] == "CP"
        dispositions2 = (
            data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(
                tic_id=25132999
            )
        )
        assert dispositions2.shape[0] == 0

    def test_toi_file_is_not_updated_from_exofop_until_first_toi_table_access(self):
        with patch.object(module.requests, "get") as mock_get:
            data_interface = TessToiDataInterface()
            data_interface.toi_dispositions_path = MagicMock()
            data_interface.load_toi_dispositions_in_project_format = Mock()
            assert not mock_get.called
            _ = data_interface.toi_dispositions
            assert mock_get.called

    def test_ctoi_file_is_not_updated_from_exofop_until_first_ctoi_table_access(self):
        with patch.object(module.requests, "get") as mock_get:
            data_interface = TessToiDataInterface()
            data_interface.ctoi_dispositions_path = MagicMock()
            data_interface.load_ctoi_dispositions_in_project_format = Mock()
            assert not mock_get.called
            _ = data_interface.ctoi_dispositions
            assert mock_get.called
