"""
A viewer for a CSV file containing a column of paths.
"""
from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.models import BoxAnnotation, Button, Div
from bokeh.server.server import Server

from ramjet.analysis.transit_vetter import TransitVetter
from ramjet.analysis.viewer.light_curve_display import LightCurveDisplay
from ramjet.analysis.viewer.preloader import Preloader
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiColumnName, TessFfiLightCurve

if TYPE_CHECKING:
    from bokeh.document import Document

    from ramjet.analysis.viewer.view_entity import ViewEntity


class Viewer:
    """
    A viewer for a CSV file containing a column of paths.
    """

    vetter = TransitVetter()

    def __init__(self):
        self.light_curve_display: LightCurveDisplay | None = None
        self.preloader: Preloader | None = None
        self.add_to_positives_button: Button | None = None
        self.previous_button: Button | None = None
        self.next_button: Button | None = None
        self.information_div: Div | None = None
        self.document: Document | None = None
        self.maximum_physical_depth_box: BoxAnnotation | None = None
        self.view_entity: ViewEntity | None = None
        self.background_tasks = set()

    async def update_view_entity_with_document_lock(self, view_entity: ViewEntity):
        """
        Updates the light curve display using the Bokeh document lock.

        :param view_entity: The view entity to update the display with.
        """
        light_curve = view_entity.light_curve
        self.document.add_next_tick_callback(
            partial(self.light_curve_display.update_from_light_curve, light_curve=light_curve)
        )
        self.document.add_next_tick_callback(
            partial(self.update_information_div_for_view_entity, view_entity=view_entity)
        )
        self.document.add_next_tick_callback(
            partial(self.add_physical_depth_range_annotation_to_light_curve_figure, view_entity=view_entity)
        )
        self.view_entity = view_entity

    async def add_physical_depth_range_annotation_to_light_curve_figure(self, view_entity: ViewEntity):
        unknown_radius = False
        maximum_depth = self.vetter.get_maximum_physical_depth_for_planet_for_target(
            view_entity.target, allow_missing_contamination_ratio=True
        )
        if np.isnan(maximum_depth):
            maximum_depth = 0.1
            unknown_radius = True
        self.maximum_physical_depth_box.bottom = 1 - maximum_depth
        if view_entity.has_exofop_dispositions:
            self.maximum_physical_depth_box.fill_color = "red"
        elif unknown_radius:
            self.maximum_physical_depth_box.fill_color = "yellow"
        else:
            self.maximum_physical_depth_box.fill_color = "green"

    async def update_information_div_for_view_entity(self, view_entity: ViewEntity):
        self.information_div.text = (
            f'<h1 class="title">TIC {view_entity.light_curve.tic_id} '
            f"sector {view_entity.light_curve.sector}</h1>"
            f"<p>Network confidence: {view_entity.confidence}</p>"
            f"<p>Result index: {view_entity.index}</p>"
            f"<p>Star radius (solar radii): {view_entity.target.radius}</p>"
        )

    async def display_next_view_entity(self):
        """
        Moves to the next view entity.
        """
        next_view_entity = await self.preloader.increment()
        await self.update_view_entity_with_document_lock(next_view_entity)

    async def display_previous_view_entity(self):
        """
        Moves to the previous view entity.
        """
        previous_view_entity = await self.preloader.decrement()
        await self.update_view_entity_with_document_lock(previous_view_entity)

    def create_display_next_view_entity_task(self):
        """
        Creates the async task to move to the next light curve.
        """
        view_entity_task = asyncio.create_task(self.display_next_view_entity())
        self.background_tasks.add(view_entity_task)
        view_entity_task.add_done_callback(self.background_tasks.discard)

    def create_display_previous_view_entity_task(self):
        """
        Creates the async task to move to the previous light curve.
        """
        view_entity_task = asyncio.create_task(self.display_previous_view_entity())
        self.background_tasks.add(view_entity_task)
        view_entity_task.add_done_callback(self.background_tasks.discard)

    def create_light_curve_switching_buttons(self) -> (Button, Button):
        """
        Creates buttons for switching between light curves.
        """
        next_button = Button(label="Next target")
        next_button.on_click(self.create_display_next_view_entity_task)
        next_button.sizing_mode = "stretch_width"
        previous_button = Button(label="Previous target")
        previous_button.on_click(self.create_display_previous_view_entity_task)
        previous_button.sizing_mode = "stretch_width"
        return previous_button, next_button

    def create_add_to_positives_button(self) -> Button:
        add_to_positives_button = Button(label="Add to positives")
        add_to_positives_button.on_click(self.add_current_to_positives)
        add_to_positives_button.sizing_mode = "stretch_width"
        return add_to_positives_button

    def add_current_to_positives(self):
        positives_csv_file_path = Path("positives.csv")
        positives_data_frame = pd.DataFrame({"tic_id": [self.view_entity.target.tic_id]})
        if positives_csv_file_path.exists():
            positives_data_frame.to_csv(positives_csv_file_path, mode="a", header=False, index=False)
        else:
            positives_data_frame.to_csv(positives_csv_file_path, index=False)

    @classmethod
    def from_csv_path(cls, bokeh_document: Document, csv_path: Path) -> Viewer:
        """
        Creates a viewer from a CSV path containing a light curve path column.

        :param bokeh_document: The Bokeh document to run the viewer in.
        :param csv_path: The path to the CSV file.
        :return: The viewer.
        """
        viewer = cls()
        viewer.document = bokeh_document
        viewer.csv_path = csv_path
        viewer.light_curve_display = LightCurveDisplay.for_columns(
            TessFfiColumnName.TIME__BTJD.value, TessFfiLightCurve().flux_column_names, flux_axis_label="Relative flux"
        )
        viewer.light_curve_display.exclude_outliers_from_zoom = True
        viewer.maximum_physical_depth_box = BoxAnnotation(bottom=1 - 0.01, top=1, fill_alpha=0.1, fill_color="green")
        viewer.light_curve_display.figure.add_layout(viewer.maximum_physical_depth_box)
        viewer.add_to_positives_button = viewer.create_add_to_positives_button()
        bokeh_document.add_root(viewer.add_to_positives_button)
        viewer.previous_button, viewer.next_button = viewer.create_light_curve_switching_buttons()
        bokeh_document.add_root(viewer.previous_button)
        bokeh_document.add_root(viewer.next_button)
        viewer.information_div = Div()
        viewer.information_div.sizing_mode = "stretch_width"
        bokeh_document.add_root(viewer.information_div)
        bokeh_document.add_root(viewer.light_curve_display.figure)
        loop = asyncio.get_running_loop()
        viewer_task = loop.create_task(viewer.start_preloader(csv_path))
        viewer.background_tasks.add(viewer_task)
        viewer_task.add_done_callback(viewer.background_tasks.discard)
        return viewer

    async def start_preloader(self, csv_path):
        """
        Starts the light curve preloader.
        """
        self.preloader = await Preloader.from_csv_path(csv_path, starting_index=0)
        initial_view_entity = self.preloader.current_view_entity
        await self.update_view_entity_with_document_lock(initial_view_entity)


def application(bokeh_document: Document):
    """
    The application to run from the Tornado server.

    :param bokeh_document: The Bokeh document to run the viewer in.
    """
    csv_path = Path("/Users/golmsche/Desktop/infer results 2020-10-09-13-21-21.csv")
    Viewer.from_csv_path(bokeh_document, csv_path)


if __name__ == "__main__":
    document = curdoc()
    server = Server({"/": application}, port=5010)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
