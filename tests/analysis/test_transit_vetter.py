from unittest.mock import patch

import pytest

import ramjet.analysis.transit_vetter as module
from ramjet.analysis.transit_vetter import TransitVetter
from ramjet.photometric_database.tess_target import TessTarget


class TestTransitVetter:
    @pytest.mark.parametrize('transit_depth, target_radius, target_contamination_ratio, expected_is_physical',
                             [(0.01, 1.0, 0.01, True),
                              (0.01, 2.0, 0.01, False),
                              (0.01, 1.0, 0.75, False)])
    def test_can_check_if_depth_for_tic_id_is_physical_for_planet(self, transit_depth, target_radius,
                                                                  target_contamination_ratio, expected_is_physical):
        tic_id = 1
        stub_target = TessTarget()
        stub_target.radius = target_radius
        stub_target.contamination_ratio = target_contamination_ratio
        transit_vetter = TransitVetter()
        with patch.object(module.TessTarget, 'from_tic_id') as stub_target_from_tic_id:
            stub_target_from_tic_id.return_value = stub_target

            is_physical = transit_vetter.is_transit_depth_for_tic_id_physical_for_planet(transit_depth=transit_depth,
                                                                                         tic_id=tic_id)

        assert is_physical == expected_is_physical
