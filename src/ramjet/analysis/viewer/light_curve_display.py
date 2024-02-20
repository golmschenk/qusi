"""
Code for displaying a light curve figure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, CustomJS, GlyphRenderer
from bokeh.plotting import figure as Figure

from ramjet.analysis.color_palette import light_curve_colors
from ramjet.analysis.convert_column_name_to_display_name import convert_column_name_to_display_name
from ramjet.analysis.light_curve_visualizer import calculate_inlier_range

if TYPE_CHECKING:
    from bokeh.colors import Color

    from ramjet.photometric_database.light_curve import LightCurve


class LightCurveDisplay:
    """
    A class for displaying a light curve figure.
    """

    def __init__(self):
        self.figure: Figure | None = None
        self.data_source: ColumnDataSource | None = None
        self.time_column_name: str | None = None
        self.flux_column_names: list[str] = []
        self.inlier_range_data_source: ColumnDataSource | None = None
        self.inlier_range_glyph_renderer: GlyphRenderer | None = None
        self.exclude_outliers_from_zoom: bool = False

    @classmethod
    def for_columns(cls, time_column_name: str, flux_column_names: list[str], flux_axis_label: str = "Flux"):
        """
        Creates a light curve display object with the specified columns prepared for display.

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
        display.initialize_data_source(column_names=[time_column_name, *flux_column_names])
        for flux_column_name, color in zip(display.flux_column_names, light_curve_colors):
            display.add_flux_data_source_line_to_figure(
                time_column_name=time_column_name, flux_column_name=flux_column_name, color=color
            )
        return display

    def initialize_figure(self, time_axis_label: str, flux_axis_label: str):
        """
        Initializes the figure.

        :param time_axis_label: The time axis label.
        :param flux_axis_label: The flux axis label.
        """
        self.figure = Figure(
            x_axis_label=time_axis_label,
            y_axis_label=flux_axis_label,
            active_drag="box_zoom",
            active_scroll="wheel_zoom",
        )
        self.figure.sizing_mode = "stretch_width"

    def initialize_data_source(self, column_names: list[str]):
        """
        Creates a data source with the passed column names.

        :param column_names: The column names to include in the data source.
        """
        self.data_source = ColumnDataSource(data=pd.DataFrame({column_name: [] for column_name in column_names}))
        js_reset = CustomJS(args={"figure": self.figure}, code="figure.reset.emit()")
        self.data_source.js_on_change("data", js_reset)

    def add_flux_data_source_line_to_figure(self, time_column_name: str, flux_column_name: str, color: Color):
        """
        Add a flux data source time series line to the figure.

        :param time_column_name: The name to use for the time column.
        :param flux_column_name: The name to use for the flux column.
        :param color: The color to use for the line.
        """
        legend_label = convert_column_name_to_display_name(flux_column_name)
        self.figure.line(
            x=time_column_name, y=flux_column_name, source=self.data_source, line_color=color, line_alpha=0.1
        )
        self.figure.circle(
            x=time_column_name,
            y=flux_column_name,
            source=self.data_source,
            legend_label=legend_label,
            line_color=color,
            line_alpha=0.4,
            fill_color=color,
            fill_alpha=0.1,
        )

    async def update_from_light_curve(self, light_curve: LightCurve):
        """
        Update the data for the display based on a light curve.

        :param light_curve: The light curve to display.
        """
        self.data_source.data = light_curve.data_frame
        if self.exclude_outliers_from_zoom:
            y_range_minimum, y_range_maximum = await calculate_inlier_range(light_curve.fluxes)
            await self.set_view_ranges(
                x_range=(np.nanmin(light_curve.times), np.nanmax(light_curve.times)),
                y_range=(y_range_minimum, y_range_maximum),
            )

    async def set_view_ranges(self, x_range: (float, float), y_range: (float, float)):
        """
        Sets the view range to a specified range.

        :param x_range: The start and end of the x range.
        :param y_range: The start and end of the y range.
        """
        if self.inlier_range_glyph_renderer is None:
            await self.enable_auto_range_selection()
        data_frame = pd.DataFrame({"x": x_range, "y": y_range})
        self.inlier_range_data_source.data = data_frame

    async def enable_auto_range_selection(self):
        """
        Enables selecting a specific auto range by creating a dummy invisible glyph and auto ranging to that.
        """
        initial_data_frame = pd.DataFrame({"x": [], "y": []})
        self.inlier_range_data_source = ColumnDataSource(data=initial_data_frame)
        self.inlier_range_glyph_renderer = self.figure.circle(
            x="x", y="y", source=self.inlier_range_data_source, alpha=0
        )
        self.figure.y_range.renderers = [self.inlier_range_glyph_renderer]
