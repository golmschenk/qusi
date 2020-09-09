"""
A viewer for a CSV file containing a column of paths.
"""
from __future__ import annotations
import asyncio
from functools import partial

import pandas as pd
from pathlib import Path
from typing import Union

from bokeh.document import Document
from bokeh.io import curdoc
from bokeh.models import Button, Div
from bokeh.server.server import Server

from ramjet.analysis.viewer.light_curve_display import LightCurveDisplay
from ramjet.analysis.viewer.preloader import Preloader
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessTwoMinuteCadenceLightCurve, \
    TessTwoMinuteCadenceColumnName


class CsvPathViewer:
    def __init__(self):
        self.light_curve_display: Union[LightCurveDisplay, None] = None
        self.preloader: Union[Preloader, None] = None
        self.previous_button: Union[Button, None] = None
        self.next_button: Union[Button, None] = None
        self.information_div: Union[Div, None] = None
        self.csv_path: Union[Path, None] = None
        self.document: Union[Document, None] = None

    async def update_light_curve_with_document_lock(self, light_curve):
        self.document.add_next_tick_callback(partial(self.light_curve_display.update_from_light_curve,
                                                     lightcurve=light_curve))

    async def display_next_target(self):
        next_index_light_curve_pair = await self.preloader.increment()
        next_light_curve = next_index_light_curve_pair.light_curve
        await self.update_light_curve_with_document_lock(next_light_curve)

    async def display_previous_target(self):
        previous_index_light_curve_pair = await self.preloader.decrement()
        previous_light_curve = previous_index_light_curve_pair.light_curve
        await self.update_light_curve_with_document_lock(previous_light_curve)

    def create_display_next_target_task(self):
        asyncio.create_task(self.display_next_target())

    def create_display_previous_target_task(self):
        asyncio.create_task(self.display_previous_target())

    def create_target_switching_buttons(self) -> (Button, Button):
        next_button = Button(label='Next target')
        next_button.on_click(self.create_display_next_target_task)
        next_button.sizing_mode = 'stretch_width'
        previous_button = Button(label='Previous target')
        previous_button.on_click(self.create_display_previous_target_task)
        previous_button.sizing_mode = 'stretch_width'
        return previous_button, next_button

    @classmethod
    def from_csv_path(cls, bokeh_document: Document, csv_path: Path) -> CsvPathViewer:
        viewer = cls()
        viewer.document = bokeh_document
        viewer.csv_path = csv_path
        viewer.light_curve_display = LightCurveDisplay.for_columns(TessTwoMinuteCadenceColumnName.TIME.value,
                                                                   TessTwoMinuteCadenceLightCurve.flux_column_names,
                                                                   flux_axis_label='Relative flux')
        viewer.previous_button, viewer.next_button = viewer.create_target_switching_buttons()
        bokeh_document.add_root(viewer.previous_button)
        bokeh_document.add_root(viewer.next_button)
        # bokeh_document.add_root(viewer.information_div)
        bokeh_document.add_root(viewer.light_curve_display.figure)
        return viewer

    async def start_preloader(self):
        paths_data_frame = pd.read_csv(self.csv_path)
        paths = paths_data_frame['Light curve path']
        self.preloader = await Preloader.from_light_curve_path_list(paths)
        initial_light_curve = self.preloader.current_index_light_curve_pair.light_curve
        await self.update_light_curve_with_document_lock(initial_light_curve)


def application(bokeh_document: Document):
    csv_path = Path('/Users/golmschenk/Code/ramjet/data/viewer_check.csv')
    viewer = CsvPathViewer.from_csv_path(bokeh_document, csv_path)
    loop = asyncio.get_running_loop()
    loop.create_task(viewer.start_preloader())


if __name__ == '__main__':
    document = curdoc()
    server = Server({'/': application}, port=5009)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
