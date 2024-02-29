import numpy as np
import pandas as pd
import pytest

from ramjet.analysis.transit_vetter import TransitVetter
from ramjet.photometric_database.tess_target import TessTarget


class TestTransitVetter:
    @pytest.mark.parametrize(
        (
            "transit_depth",
            "target_radius",
            "target_contamination_ratio",
            "expected_is_physical",
        ),
        [(0.01, 1.0, 0.01, True), (0.01, 2.0, 0.01, False), (0.01, 1.0, 5, False)],
    )
    def test_can_check_if_depth_for_tic_id_is_physical_for_planet(
        self,
        transit_depth,
        target_radius,
        target_contamination_ratio,
        expected_is_physical,
    ):
        stub_target = TessTarget()
        stub_target.tic_id = 1
        stub_target.radius = target_radius
        stub_target.contamination_ratio = target_contamination_ratio
        transit_vetter = TransitVetter()

        is_physical = transit_vetter.is_transit_depth_for_target_physical_for_planet(
            target=stub_target, transit_depth=transit_depth
        )

        assert is_physical == expected_is_physical

    @pytest.mark.parametrize(
        ("nearby_separations", "nearby_magnitudes", "expected_ruling"),
        [
            ([30], [12], True),
            ([5], [12], False),
            ([5], [18], True),
            ([50, 1], [18, 12], False),
        ],
    )
    def test_can_check_if_nearby_targets_might_be_background_eclipsing_binary(
        self, nearby_separations, nearby_magnitudes, expected_ruling
    ):
        stub_target = TessTarget()
        stub_target.magnitude = 10
        stub_target.retrieve_nearby_tic_targets = lambda: pd.DataFrame(
            {"Separation (arcsec)": nearby_separations, "TESS Mag": nearby_magnitudes}
        )
        transit_vetter = TransitVetter()

        has_problematic_nearby_targets = (
            transit_vetter.has_no_nearby_likely_eclipsing_binary_background_targets(
                stub_target
            )
        )

        assert has_problematic_nearby_targets == expected_ruling

    @pytest.mark.parametrize(
        ("nearby_toi", "nearby_separations", "expected_ruling"),
        [([1], [50], False), ([1], [10], True), ([1, 2], [50, 10], True)],
    )
    def test_can_check_if_nearby_targets_are_known_toi_targets(
        self, nearby_toi, nearby_separations, expected_ruling
    ):
        stub_target = TessTarget()
        stub_target.retrieve_nearby_tic_targets = lambda: pd.DataFrame(
            {"Separation (arcsec)": nearby_separations, "TOI": nearby_toi}
        )
        transit_vetter = TransitVetter()

        has_problematic_nearby_targets = transit_vetter.has_nearby_toi_targets(
            stub_target
        )

        assert has_problematic_nearby_targets == expected_ruling

    @pytest.mark.parametrize(
        ("target_radius", "target_contamination_ratio", "expected_maximum_depth"),
        [(1.0, 0.0, 0.0342398016), (3.0, 0.5, 0.0025362816)],
    )
    def test_maximum_depth_for_planet(
        self, target_radius, target_contamination_ratio, expected_maximum_depth
    ):
        stub_target = TessTarget()
        stub_target.tic_id = 1
        stub_target.radius = target_radius
        stub_target.contamination_ratio = target_contamination_ratio
        transit_vetter = TransitVetter()

        maximum_depth = transit_vetter.get_maximum_physical_depth_for_planet_for_target(
            target=stub_target
        )

        assert maximum_depth == pytest.approx(expected_maximum_depth)

    @pytest.mark.parametrize(
        ("target_radius", "target_contamination_ratio", "expected_maximum_depth"),
        [(1.0, np.nan, 0.0342398016)],
    )
    def test_max_depth_for_planet_uses_zero_contamination_if_contamination_is_not_available(
        self, target_radius, target_contamination_ratio, expected_maximum_depth
    ):
        stub_target = TessTarget()
        stub_target.tic_id = 1
        stub_target.radius = target_radius
        stub_target.contamination_ratio = target_contamination_ratio
        transit_vetter = TransitVetter()

        maximum_depth0 = (
            transit_vetter.get_maximum_physical_depth_for_planet_for_target(
                target=stub_target, allow_missing_contamination_ratio=True
            )
        )

        assert maximum_depth0 == pytest.approx(expected_maximum_depth)

        with pytest.raises(
            ValueError, match="Contamination ratio nan is not a number."
        ):
            _ = transit_vetter.get_maximum_physical_depth_for_planet_for_target(
                target=stub_target
            )
