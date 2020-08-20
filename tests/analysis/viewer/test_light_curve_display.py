from ramjet.analysis.viewer.light_curve_display import LightCurveDisplay


class TestLightCurveDisplay:
    def test_for_columns_factory_initializes_figure_with_human_readable_time_axis_label(self):
        display = LightCurveDisplay.for_columns(time_column_name='time__days', flux_column_names=['pdcsap', 'sap'])
        assert display.figure.xaxis.axis_label == 'Time (days)'
