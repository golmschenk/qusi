"""
Code for displaying a light curve figure.
"""
from typing import Union, List

from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure

from ramjet.analysis.viewer.convert_column_name_to_display_name import convert_column_name_to_display_name


class LightCurveDisplay:
    """
    A class for displaying a light curve figure.
    """
    def __init__(self):
        self.figure: Union[Figure, None] = None
        self.data_source: Union[ColumnDataSource, None] = None
        self.time_column_name: Union[str, None] = None
        self.flux_column_names: List[str] = []

    @classmethod
    def for_columns(cls, time_column_name: str, flux_column_names: List[str], flux_axis_label: str = 'Flux'):
        """
        Creates a lightcurve display object with the specified columns prepared for display.

        :param time_column_name: The name of the time column in the data.
        :param flux_column_names: The names of the flux columns in the data.
        :param flux_axis_label: The name to display on the flux axis.
        :return: The light curve display.
        """
        display = cls()
        display.time_column_name = time_column_name
        display.flux_column_names = flux_column_names
        time_axis_label = convert_column_name_to_display_name(time_column_name)
        display.initialize_figure(time_axis_label=time_axis_label, flux_axis_label=flux_axis_label)
        return display

    def initialize_figure(self, time_axis_label: str, flux_axis_label: str):
        """
        Initializes the figure.

        :param time_axis_label: The time axis label.
        :param flux_axis_label: The flux axis label.
        """
        self.figure = Figure(x_axis_label=time_axis_label, y_axis_label=flux_axis_label,
                             active_drag='box_zoom', active_scroll='wheel_zoom')
