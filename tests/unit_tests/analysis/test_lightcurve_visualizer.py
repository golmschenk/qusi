from unittest.mock import patch

import numpy as np

import ramjet
from ramjet.analysis.light_curve_visualizer import create_dual_light_curve_figure


def test_create_dual_light_curve_figure_normalizes_light_curve():
    fluxes0 = [0, 1, 2]
    fluxes1 = [3, 4, 5]
    times0 = [0, 10, 20]
    times1 = [30, 40, 50]
    with patch.object(
        ramjet.analysis.light_curve_visualizer.Figure, "line"
    ) as mock_line:
        _ = create_dual_light_curve_figure(
            fluxes0, times0, "name0", fluxes1, times1, "name1", "title"
        )
        # [0] is the first call, [0][0] is the args of the first call, [0][0][1] is second arg of first call.
        assert np.array_equal(
            mock_line.call_args_list[0][0][1], [0, 1, 2]
        )  # Median is 1 so it doesn't change.
        assert np.array_equal(
            mock_line.call_args_list[1][0][1], [0.75, 1, 1.25]
        )  # Median is 4.


def test_create_dual_light_curve_figure_does_not_invert_negative_light_curve():
    fluxes0 = [0, -1, -2]
    fluxes1 = [-3, -4, -5]
    times0 = [0, 10, 20]
    times1 = [30, 40, 50]
    with patch.object(
        ramjet.analysis.light_curve_visualizer.Figure, "line"
    ) as mock_line:
        _ = create_dual_light_curve_figure(
            fluxes0, times0, "name0", fluxes1, times1, "name1", "title"
        )
        # This expects the minimum value is subtracted from the fluxes if it's less than zero, before normalization.
        # [0] is the first call, [0][0] is the args of the first call, [0][0][1] is second arg of first call.
        assert np.array_equal(mock_line.call_args_list[0][0][1], [2, 1, 0])
        assert np.array_equal(mock_line.call_args_list[1][0][1], [2, 1, 0])
