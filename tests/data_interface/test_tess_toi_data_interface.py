import pytest
import numpy as np

from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface


class TestTessToiDataInterface:
    @pytest.fixture
    def data_interface(self) -> TessToiDataInterface:
        return TessToiDataInterface()

    @pytest.mark.slow
    @pytest.mark.external
    def test_can_retrieve_the_tess_toi_dispositions(self, data_interface):
        dispositions = data_interface.dispositions
        target_dispositions = dispositions[dispositions['TIC ID'] == 307210830]
        assert target_dispositions['Disposition'].iloc[0] == 'CP'
        target_planet_sectors = target_dispositions['Sector'].unique()
        assert np.array_equal(np.sort(np.array(target_planet_sectors)), [2, 5, 8, 9, 10, 11, 12])
