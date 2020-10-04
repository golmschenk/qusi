from unittest.mock import Mock

import numpy as np
from collections import defaultdict

import pytest

from ramjet.photometric_database.tess_target import TessTarget


class TestTessTarget:
    def test_from_tic_id_uses_gaia_radius_if_tic_radius_is_nan(self):
        stub_tic_row = defaultdict(int)
        stub_tic_row['rad'] = np.nan
        stub_tic_row['GAIA'] = 1
        TessTarget.tess_data_interface.get_tess_input_catalog_row = Mock(return_value=stub_tic_row)
        mock_gaia_mass = Mock()
        TessTarget.get_radius_from_gaia = Mock(return_value=mock_gaia_mass)

        target = TessTarget.from_tic_id(1)

        assert target.radius == mock_gaia_mass

    @pytest.mark.external
    def test_retrieving_radius_from_gaia(self):
        target = TessTarget()

        gaia_radius = target.get_radius_from_gaia(2057537107164000640)

        assert gaia_radius == pytest.approx(4.2563343)
