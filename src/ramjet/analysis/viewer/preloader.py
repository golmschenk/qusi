"""
Code to load view entities in the background so they show up quickly when displayed.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from asyncio import Task
from collections import deque
from typing import TYPE_CHECKING

import pandas as pd

from ramjet.analysis.viewer.view_entity import ViewEntity
from ramjet.data_interface.tess_data_interface import NoDataProductsFoundError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class Preloader:
    """
    A class to load view entities in the background so they show up quickly when displayed.
    """

    minimum_preloaded = 25
    maximum_preloaded = 50

    def __init__(self):
        self.current_view_entity: None | ViewEntity = None
        self.next_view_entity_deque: deque[ViewEntity] = deque(maxlen=self.maximum_preloaded)
        self.previous_view_entity_deque: deque[ViewEntity] = deque(maxlen=self.maximum_preloaded)
        self.identifier_data_frame: pd.DataFrame | None = None
        self.running_loading_task: Task | None = None

    async def load_view_entity_at_index_as_current(self, index: int):
        """
        Loads the view entity at the passed index as the current view entity.

        :param index: The index in the path list to load.
        """
        await self.cancel_loading_task()
        self.current_view_entity = await ViewEntity.from_identifier_data_frame_row(
            self.identifier_data_frame.iloc[index]
        )
        await self.reset_deques()

    async def load_surrounding_view_entities(self):
        """
        Loads the next and previous view entities relative to the current view entity.
        """
        await self.load_next_view_entities()
        await self.load_previous_view_entities()

    async def load_next_view_entities(self):
        """
        Preload the next view entities.
        """
        if len(self.next_view_entity_deque) > 0:
            last_index = self.next_view_entity_deque[-1].index
        else:
            last_index = self.current_view_entity.index
        while (
            len(self.next_view_entity_deque) < self.minimum_preloaded
            and last_index != self.identifier_data_frame.shape[0] - 1
        ):
            last_index += 1
            try:
                last_view_entity = await ViewEntity.from_identifier_data_frame_row(
                    self.identifier_data_frame.iloc[last_index]
                )
            except NoDataProductsFoundError:
                logger.warning(f"No light curve found for identifier {self.identifier_data_frame.iloc[last_index]}.")
                continue
            self.next_view_entity_deque.append(last_view_entity)

    async def load_previous_view_entities(self):
        """
        Preload the previous view entities.
        """
        if len(self.previous_view_entity_deque) > 0:
            first_index = self.previous_view_entity_deque[0].index
        else:
            first_index = self.current_view_entity.index
        while len(self.previous_view_entity_deque) < self.minimum_preloaded and first_index != 0:
            first_index -= 1
            try:
                first_view_entity = await ViewEntity.from_identifier_data_frame_row(
                    self.identifier_data_frame.iloc[first_index]
                )
            except NoDataProductsFoundError:
                logger.warning(f"No light curve found for identifier {self.identifier_data_frame.iloc[first_index]}.")
                continue
            self.previous_view_entity_deque.appendleft(first_view_entity)

    async def increment(self) -> ViewEntity:
        """
        Increments to the next view entity, and calls loading as necessary.

        :return: The new current view entity.
        """
        self.previous_view_entity_deque.append(self.current_view_entity)
        if len(self.next_view_entity_deque) == 0 and (
            self.running_loading_task is not None and not self.running_loading_task.done()
        ):
            await self.running_loading_task
        self.current_view_entity = self.next_view_entity_deque.popleft()
        await self.refresh_surrounding_light_curve_loading()
        return self.current_view_entity

    async def decrement(self) -> ViewEntity:
        """
        Decrements to the previous view entity, and calls loading as necessary.

        :return: The new current view entity.
        """
        self.next_view_entity_deque.appendleft(self.current_view_entity)
        while len(self.previous_view_entity_deque) == 0 and (
            self.running_loading_task is not None and not self.running_loading_task.done()
        ):
            await self.running_loading_task
        self.current_view_entity = self.previous_view_entity_deque.pop()
        await self.refresh_surrounding_light_curve_loading()
        return self.current_view_entity

    async def refresh_surrounding_light_curve_loading(self):
        """
        Cancels the existing loading task and starts a new one.
        """
        await self.cancel_loading_task()
        self.running_loading_task = asyncio.create_task(self.load_surrounding_view_entities())

    async def cancel_loading_task(self):
        """
        Cancels an existing loading task if it exists.
        """
        if self.running_loading_task is not None:
            self.running_loading_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.running_loading_task

    async def reset_deques(self):
        """
        Cancels any loading tasks, clears the deques, and starts the loading task.
        """
        await self.cancel_loading_task()
        self.previous_view_entity_deque = deque(maxlen=self.maximum_preloaded)
        self.next_view_entity_deque = deque(maxlen=self.maximum_preloaded)
        self.running_loading_task = asyncio.create_task(self.load_surrounding_view_entities())

    @classmethod
    async def from_csv_path(cls, csv_path: Path, starting_index: int = 0):
        """
        Create a preloader from a CSV of light curve identifiers.

        :param csv_path: A path to a CSV containing light curve identifier information.
        :param starting_index: The starting index to preload around.
        :return: The preloader.
        """
        preloader = cls()
        preloader.identifier_data_frame = pd.read_csv(csv_path)
        await preloader.load_view_entity_at_index_as_current(starting_index)
        return preloader
