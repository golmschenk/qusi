from unittest.mock import Mock

import pytest
import numpy as np
import pandas as pd

from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface


class TestTessToiDataInterface:
    @pytest.fixture
    def data_interface(self) -> TessToiDataInterface:
        return TessToiDataInterface()

    @pytest.mark.slow
    @pytest.mark.external
    def test_can_retrieve_the_tess_toi_dispositions(self, data_interface):
        dispositions = data_interface.toi_dispositions
        target_dispositions = dispositions[dispositions['TIC ID'] == 307210830]
        assert target_dispositions['Disposition'].iloc[0] == 'CP'
        target_planet_sectors = target_dispositions['Sector'].unique()
        assert np.array_equal(np.sort(np.array(target_planet_sectors)), [2, 5, 8, 9, 10, 11, 12])

    def test_can_get_exofop_planet_disposition_for_tic_id(self, data_interface):
        mock_dispositions = pd.DataFrame({'TIC ID': [231663901, 266980320], 'Disposition': ['KP', 'CP']})
        data_interface.retrieve_toi_dispositions_from_exofop = Mock(return_value=mock_dispositions)
        dispositions0 = data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(tic_id=231663901)
        assert dispositions0['Disposition'].iloc[0] == 'KP'
        dispositions1 = data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(tic_id=266980320)
        assert dispositions1['Disposition'].iloc[0] == 'CP'
        dispositions2 = data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(tic_id=25132999)
        assert dispositions2.shape[0] == 0