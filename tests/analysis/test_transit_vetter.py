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
        stub_target = TessTarget()
        stub_target.tic_id = 1
        stub_target.radius = target_radius
        stub_target.contamination_ratio = target_contamination_ratio
        transit_vetter = TransitVetter()

        is_physical = transit_vetter.is_transit_depth_for_target_physical_for_planet(target=stub_target,
                                                                                     transit_depth=transit_depth)

        assert is_physical == expected_is_physical

    def test_can_check_if_nearby_targets_could_be_troublesome(self):
        stub_target = TessTarget()
        stub_target.tic_id = 1
