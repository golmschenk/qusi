"""
Code for displaying a light curve figure.
"""
import pandas as pd
from typing import Union, List

from bokeh.colors import Color
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure

from ramjet.analysis.color_palette import lightcurve_colors
from ramjet.analysis.viewer.convert_column_name_to_display_name import convert_column_name_to_display_name
from ramjet.photometric_database.light_curve import LightCurve


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
        display.initialize_data_source(column_names=[time_column_name] + flux_column_names)
        for flux_column_name, color in zip(display.flux_column_names, lightcurve_colors):
            display.add_flux_data_source_line_to_figure(time_column_name=time_column_name, flux_column_name=flux_column_name,
                                                        color=color)
        return display

    def initialize_figure(self, time_axis_label: str, flux_axis_label: str):
        """
        Initializes the figure.

        :param time_axis_label: The time axis label.
        :param flux_axis_label: The flux axis label.
        """
        self.figure = Figure(x_axis_label=time_axis_label, y_axis_label=flux_axis_label,
                             active_drag='box_zoom', active_scroll='wheel_zoom')
        self.figure.sizing_mode = 'stretch_width'

    def initialize_data_source(self, column_names: List[str]):
        """
        Creates a data source with the passed column names.

        :param column_names: The column names to include in the data source.
        """
        self.data_source = ColumnDataSource(data=pd.DataFrame({column_name: [] for column_name in column_names}))

    def add_flux_data_source_line_to_figure(self, time_column_name: str, flux_column_name: str, color: Color):
        """
        Add a flux data source time series line to the figure.

        :param time_column_name: The name to use for the time column.
        :param flux_column_name: The name to use for the flux column.
        :param color: The color to use for the line.
        """
        legend_label = convert_column_name_to_display_name(flux_column_name)
        self.figure.line(x=time_column_name, y=flux_column_name, source=self.data_source, line_color=color,
                         line_alpha=0.1)
        self.figure.circle(x=time_column_name, y=flux_column_name, source=self.data_source, legend_label=legend_label,
                           line_color=color, line_alpha=0.4, fill_color=color, fill_alpha=0.1)

    async def update_from_light_curve(self, lightcurve: LightCurve):
        """
        Update the data for the display based on a light curve.

        :param lightcurve: The light curve to display.
        """
        self.data_source.data = lightcurve.data_frame
