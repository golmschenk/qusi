from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import units
from bokeh.events import Tap
from bokeh.palettes import Turbo256
from lightkurve.periodogram import LombScarglePeriodogram, Periodogram

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from bokeh.models import ColumnDataSource, Div, LinearAxis, LinearColorMapper, Range1d, Span, Spinner, TapTool
from bokeh.plotting import figure as Figure

if TYPE_CHECKING:
    from bokeh.document import Document

    from ramjet.photometric_database.light_curve import LightCurve


class FoldedLightCurveColumnName(StrEnum):
    TIME = "time"
    FOLDED_TIME = "folded_time"
    FLUX = "flux"


class PeriodogramColumnName(StrEnum):
    PERIOD = "period"
    POWER = "power"


class Viewer:
    def __init__(self, bokeh_document: Document, light_curve: LightCurve, title: str | None = None):
        self.bokeh_document: Document = bokeh_document
        tool_tips = [
            ("Time", f"@{FoldedLightCurveColumnName.TIME}{{0.0000000}}"),
            ("Folded time", f"@{FoldedLightCurveColumnName.FOLDED_TIME}{{0.0000000}}"),
            ("Flux", f"@{FoldedLightCurveColumnName.FLUX}{{0.0000000}}"),
        ]
        self.folded_light_curve_figure: Figure = Figure(tooltips=tool_tips)
        self.folded_light_curve_figure.sizing_mode = "stretch_width"
        self.unfolded_light_curve_figure: Figure = Figure(tooltips=tool_tips)
        self.unfolded_light_curve_figure.sizing_mode = "stretch_width"
        self.light_curve: LightCurve = light_curve
        flux_median = np.nanmedian(self.light_curve.fluxes)
        fluxes = self.light_curve.fluxes
        relative_fluxes = fluxes / flux_median
        minimum_time = np.nanmin(self.light_curve.times)
        maximum_time = np.nanmax(self.light_curve.times)
        time_differences = np.diff(self.light_curve.times)
        minimum_time_step = np.nanmin(time_differences)
        median_time_step = np.nanmedian(time_differences)
        period_upper_limit = maximum_time - minimum_time
        period_lower_limit = minimum_time_step / 2.1
        self.fold_period_spinner: Spinner = Spinner(
            value=period_upper_limit, low=period_lower_limit, high=period_upper_limit, step=median_time_step / 1000
        )
        self.fold_period_spinner.on_change("value", self.update_fold)
        mapper = LinearColorMapper(palette=Turbo256, low=minimum_time, high=maximum_time)
        color = {"field": FoldedLightCurveColumnName.TIME, "transform": mapper}

        self.unfolded_light_curve_column_data_source: ColumnDataSource = ColumnDataSource(
            data={
                FoldedLightCurveColumnName.TIME: self.light_curve.times,
                FoldedLightCurveColumnName.FOLDED_TIME: self.light_curve.times,
                FoldedLightCurveColumnName.FLUX: relative_fluxes,
            }
        )
        self.unfolded_light_curve_figure.circle(
            source=self.unfolded_light_curve_column_data_source,
            x=FoldedLightCurveColumnName.FOLDED_TIME,
            y=FoldedLightCurveColumnName.FLUX,
            line_color=color,
            line_alpha=0.8,
            fill_color=color,
            fill_alpha=0.2,
        )

        self.folded_light_curve_column_data_source: ColumnDataSource = ColumnDataSource(
            data={
                FoldedLightCurveColumnName.TIME: self.light_curve.times,
                FoldedLightCurveColumnName.FOLDED_TIME: self.light_curve.times,
                FoldedLightCurveColumnName.FLUX: relative_fluxes,
            }
        )
        self.folded_light_curve_figure.circle(
            source=self.folded_light_curve_column_data_source,
            x=FoldedLightCurveColumnName.FOLDED_TIME,
            y=FoldedLightCurveColumnName.FLUX,
            line_color=color,
            line_alpha=0.8,
            fill_color=color,
            fill_alpha=0.2,
        )

        fluxes_minimum = fluxes.min()
        fluxes_maximum = fluxes.max()
        fluxes_margin = (fluxes_maximum - fluxes_minimum) * 0.05
        fluxes_range = Range1d(fluxes_minimum - fluxes_margin, fluxes_maximum + fluxes_margin)

        relative_fluxes_minimum = relative_fluxes.min()
        relative_fluxes_maximum = relative_fluxes.max()
        relative_fluxes_margin = (relative_fluxes_maximum - relative_fluxes_minimum) * 0.05
        relative_fluxes_range = Range1d(
            relative_fluxes_minimum - relative_fluxes_margin, relative_fluxes_maximum + relative_fluxes_margin
        )

        self.folded_light_curve_figure.y_range = relative_fluxes_range
        self.folded_light_curve_figure.extra_y_ranges = {"unnormalized_range": fluxes_range}
        self.folded_light_curve_figure.add_layout(LinearAxis(y_range_name="unnormalized_range"), "right")

        self.periodogram_figure: Figure = Figure(active_drag="box_zoom")
        self.current_fold_period_span = Span(
            location=self.fold_period_spinner.value, dimension="height", line_color="firebrick"
        )
        self.periodogram_figure.add_layout(self.current_fold_period_span)
        self.periodogram_figure.add_tools(TapTool())
        self.periodogram_figure.on_event(Tap, self.periodogram_tap_callback)
        lightkurve_light_curve = self.light_curve.to_lightkurve()
        inlier_lightkurve_light_curve = lightkurve_light_curve.remove_outliers(sigma=3)
        periodogram: Periodogram = LombScarglePeriodogram.from_lightcurve(
            inlier_lightkurve_light_curve,
            oversample_factor=5,
            minimum_period=period_lower_limit,
            maximum_period=period_upper_limit,
        )
        periods__days = periodogram.period.to(units.day).value
        powers = periodogram.power.value

        self.periodogram_column_data_source: ColumnDataSource = ColumnDataSource(
            data={
                PeriodogramColumnName.PERIOD: periods__days,
                PeriodogramColumnName.POWER: powers,
            }
        )
        self.periodogram_figure.line(
            source=self.periodogram_column_data_source, x=PeriodogramColumnName.PERIOD, y=PeriodogramColumnName.POWER
        )
        self.periodogram_figure.sizing_mode = "stretch_width"

        if title is not None:
            title_div = Div(text=f"<h1>{title}</h1>")
            self.bokeh_document.add_root(title_div)
        self.bokeh_document.add_root(self.unfolded_light_curve_figure)
        self.bokeh_document.add_root(self.folded_light_curve_figure)
        self.bokeh_document.add_root(self.fold_period_spinner)
        self.bokeh_document.add_root(self.periodogram_figure)

        # TODO: Remove spike.
        # Spike to make the default fold and zoom be useful for the short period application.
        longest_period_index_near_max_power = np.argwhere(powers > 0.9 * periodogram.max_power)[0, -1]
        while powers[longest_period_index_near_max_power + 1] > powers[longest_period_index_near_max_power]:
            longest_period_index_near_max_power += 1
        longest_period_near_max_power = periods__days[longest_period_index_near_max_power]
        self.fold_period_spinner.value = longest_period_near_max_power
        self.periodogram_figure.x_range.start = period_lower_limit
        self.periodogram_figure.x_range.end = longest_period_near_max_power * 2
        self.update_view_with_new_fold()

    def update_fold(self):
        self.calculate_new_fold()
        self.update_view_with_new_fold()

    def calculate_new_fold(self):
        self.light_curve.fold(self.fold_period_spinner.value, epoch=0)

    def update_view_with_new_fold(self):
        self.current_fold_period_span.location = self.fold_period_spinner.value
        self.folded_light_curve_column_data_source.data[
            FoldedLightCurveColumnName.FOLDED_TIME
        ] = self.light_curve.folded_times

    def periodogram_tap_callback(self, event):
        self.fold_period_spinner.value = event.x
        self.update_view_with_new_fold()
