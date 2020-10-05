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

    @pytest.mark.parametrize('transit_depth, target_radius, target_contamination_ratio, expected_body_radius',
                             [(0.01011, 1.0, 0.0, 0.1005484),
                              (0.02, 1.0, 0.1, 0.1490712),
                              (0.01, 2.0, 0.5, 0.2828427)])
    def test_can_estimate_radius_of_transiting_body(self, transit_depth, target_radius, target_contamination_ratio,
                                                    expected_body_radius):
        target = TessTarget()
        target.radius = target_radius
        target.contamination_ratio = target_contamination_ratio

        body_radius = target.calculate_transiting_body_radius(transit_depth=transit_depth)

        assert body_radius == pytest.approx(expected_body_radius)

    @pytest.mark.external
    def test_can_retrieve_nearby_tic_target_data_frame(self):
        # Information from https://exofop.ipac.caltech.edu/tess/nearbytarget.php?id=231663901
        target = TessTarget()
        target.tic_id = 231663901

        nearby_target_data_frame = target.retrieve_nearby_tic_targets()

        assert 1989124451 in nearby_target_data_frame['TIC ID'].values
        assert 1989124456 in nearby_target_data_frame['TIC ID'].values
        assert 231663902 in nearby_target_data_frame['TIC ID'].values

    @pytest.mark.external
    def test_retrieve_nearby_tic_target_data_frame_corrects_exofop_bug(self):
        # Information from https://exofop.ipac.caltech.edu/tess/nearbytarget.php?id=231663901
        # As of 2020-10-5, they incorrectly leave out the header for distance error.
        target = TessTarget()
        target.tic_id = 231663901

        nearby_target_data_frame = target.retrieve_nearby_tic_targets()

        row = nearby_target_data_frame[nearby_target_data_frame['TIC ID'] == 231663901].iloc[0]
        assert row['PM RA (mas/yr)'] == pytest.approx(12.6409)
        assert row['Separation (arcsec)'] == 0
        assert row['Distance Err (pc)'] == pytest.approx(4.411)
