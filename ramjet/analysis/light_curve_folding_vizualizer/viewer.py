from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.palettes import Turbo256

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from bokeh.document import Document
from bokeh.models import Spinner, ColumnDataSource, LinearColorMapper
from bokeh.plotting import Figure

from ramjet.photometric_database.light_curve import LightCurve


class ColumnName(StrEnum):
    TIME = 'time'
    FOLDED_TIME = 'folded_time'
    FLUX = 'flux'


class Viewer:
    def __init__(self, bokeh_document: Document, light_curve: LightCurve):
        self.bokeh_document: Document = bokeh_document
        self.folded_light_curve_figure: Figure = Figure()
        self.folded_light_curve_figure.sizing_mode = 'stretch_width'
        self.light_curve: LightCurve = light_curve
        flux_median = np.median(self.light_curve.fluxes)
        relative_fluxes = self.light_curve.fluxes / flux_median
        minimum_time = min(self.light_curve.times)
        maximum_time = max(self.light_curve.times)
        time_differences = np.diff(self.light_curve.times)
        minimum_time_step = min(time_differences)
        average_time_step = np.mean(time_differences)
        self.fold_period_spinner: Spinner = Spinner(value=maximum_time-minimum_time, low=minimum_time_step,
                                                    high=maximum_time-minimum_time, step=average_time_step / 30)
        self.fold_period_spinner.on_change('value', self.update_fold)
        self.light_curve.fluxes -= np.minimum(np.nanmin(self.light_curve.fluxes), 0)
        mapper = LinearColorMapper(palette=Turbo256, low=minimum_time, high=maximum_time)
        color = {'field': ColumnName.TIME, 'transform': mapper}
        self.viewer_column_data_source: ColumnDataSource = ColumnDataSource(data={
            ColumnName.TIME: self.light_curve.times,
            ColumnName.FOLDED_TIME: self.light_curve.times,
            ColumnName.FLUX: relative_fluxes,
        })
        self.folded_light_curve_figure.circle(source=self.viewer_column_data_source, x=ColumnName.FOLDED_TIME,
                                              y=ColumnName.FLUX, line_color=color, line_alpha=0.8,
                                              fill_color=color, fill_alpha=0.2)
        self.bokeh_document.add_root(self.folded_light_curve_figure)
        self.bokeh_document.add_root(self.fold_period_spinner)

    def update_fold(self, attr, old, new):
        self.calculate_new_fold()
        self.update_view_with_new_fold()

    def calculate_new_fold(self):
        self.light_curve.fold(self.fold_period_spinner.value, epoch=0)

    def update_view_with_new_fold(self):
        self.viewer_column_data_source.data[ColumnName.FOLDED_TIME] = self.light_curve.folded_times
