"""
Code to load light curves in the background so they show up quickly when displayed.
"""
import asyncio
from asyncio import Task
from collections import deque
import warnings
from typing import Union, List, Deque, Any

from ramjet.analysis.viewer.view_entity import ViewEntity
from ramjet.data_interface.tess_data_interface import NoDataProductsFoundException
from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessTwoMinuteCadenceLightCurve


class Preloader:
    """
    A class to load light curves in the background so they show up quickly when displayed.
    """
    minimum_preloaded = 5
    maximum_preloaded = 10

    def __init__(self):
        self.current_index_light_curve_pair: Union[None, ViewEntity] = None
        self.next_index_light_curve_pair_deque: Deque[ViewEntity] = deque(maxlen=self.maximum_preloaded)
        self.previous_index_light_curve_pair_deque: Deque[ViewEntity] = deque(maxlen=self.maximum_preloaded)
        self.light_curve_identifiers: List[Any] = []
        self.running_loading_task: Union[Task, None] = None

    async def load_light_curve_at_index_as_current(self, index: int):
        """
        Loads the light curve at the passed index as the current light curve.

        :param index: The index in the path list to load.
        """
        loop = asyncio.get_running_loop()
        await self.cancel_loading_task()
        current_light_curve = await loop.run_in_executor(None, self.load_light_curve_from_identifier,
                                                         self.light_curve_identifiers[index])
        self.current_index_light_curve_pair = ViewEntity(index, current_light_curve)
        await self.reset_deques()

    async def load_surrounding_light_curves(self):
        """
        Loads the next and previous light curves relative to the current light curve.
        """
        await self.load_next_light_curves()
        await self.load_previous_light_curves()

    async def load_next_light_curves(self):
        """
        Preload the next light curves.
        """
        loop = asyncio.get_running_loop()
        if len(self.next_index_light_curve_pair_deque) > 0:
            last_index = self.next_index_light_curve_pair_deque[-1].index
        else:
            last_index = self.current_index_light_curve_pair.index
        while (len(self.next_index_light_curve_pair_deque) < self.minimum_preloaded and
               last_index != len(self.light_curve_identifiers) - 1):
            last_index += 1
            try:
                last_light_curve = await loop.run_in_executor(None, self.load_light_curve_from_identifier,
                                                              self.light_curve_identifiers[last_index])
            except NoDataProductsFoundException:
                warnings.warn(f'No light curve found for identifier {self.light_curve_identifiers[last_index]}.')
                continue
            last_index_light_curve_pair = ViewEntity(last_index, last_light_curve)
            self.next_index_light_curve_pair_deque.append(last_index_light_curve_pair)

    async def load_previous_light_curves(self):
        """
        Preload the previous light curves.
        """
        loop = asyncio.get_running_loop()
        if len(self.previous_index_light_curve_pair_deque) > 0:
            first_index = self.previous_index_light_curve_pair_deque[0].index
        else:
            first_index = self.current_index_light_curve_pair.index
        while (len(self.previous_index_light_curve_pair_deque) < self.minimum_preloaded and
               first_index != 0):
            first_index -= 1
            try:
                first_light_curve = await loop.run_in_executor(None, self.load_light_curve_from_identifier,
                                                               self.light_curve_identifiers[first_index])
            except NoDataProductsFoundException:
                warnings.warn(f'No light curve found for identifier {self.light_curve_identifiers[first_index]}.')
                continue
            first_index_light_curve_pair = ViewEntity(first_index, first_light_curve)
            self.previous_index_light_curve_pair_deque.appendleft(first_index_light_curve_pair)

    async def increment(self) -> ViewEntity:
        """
        Increments to the next light curve, and calls loading as necessary.

        :return: The new current index and light curve pair.
        """
        self.previous_index_light_curve_pair_deque.append(self.current_index_light_curve_pair)
        if len(self.next_index_light_curve_pair_deque) == 0 and (
                self.running_loading_task is not None and not self.running_loading_task.done()):
            await self.running_loading_task
        self.current_index_light_curve_pair = self.next_index_light_curve_pair_deque.popleft()
        await self.refresh_surrounding_light_curve_loading()
        return self.current_index_light_curve_pair

    async def decrement(self) -> ViewEntity:
        """
        Decrements to the previous light curve, and calls loading as necessary.

        :return: The new current index and light curve pair.
        """
        self.next_index_light_curve_pair_deque.appendleft(self.current_index_light_curve_pair)
        while len(self.previous_index_light_curve_pair_deque) == 0 and (
                self.running_loading_task is not None and not self.running_loading_task.done()):
            await self.running_loading_task
        self.current_index_light_curve_pair = self.previous_index_light_curve_pair_deque.pop()
        await self.refresh_surrounding_light_curve_loading()
        return self.current_index_light_curve_pair

    async def refresh_surrounding_light_curve_loading(self):
        """
        Cancels the existing loading task and starts a new one.
        """
        await self.cancel_loading_task()
        self.running_loading_task = asyncio.create_task(self.load_surrounding_light_curves())

    async def cancel_loading_task(self):
        """
        Cancels an existing loading task if it exists.
        """
        if self.running_loading_task is not None:
            self.running_loading_task.cancel()
            try:
                await self.running_loading_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def load_light_curve_from_identifier(identifier: Any) -> LightCurve:
        """
        Loads a light curve from a generic identifier.

        :param identifier: The identifier of the light curve.
        :return: The light curve.
        """
        light_curve = TessTwoMinuteCadenceLightCurve.from_identifier(identifier)
        light_curve.convert_to_relative_scale()
        return light_curve

    async def reset_deques(self):
        """
        Cancels any loading tasks, clears the deques, and starts the loading task.
        """
        await self.cancel_loading_task()
        self.previous_index_light_curve_pair_deque = deque(maxlen=self.maximum_preloaded)
        self.next_index_light_curve_pair_deque = deque(maxlen=self.maximum_preloaded)
        self.running_loading_task = asyncio.create_task(self.load_surrounding_light_curves())

    @classmethod
    async def from_identifier_list(cls, identifiers: List[Any], starting_index: int = 0):
        """
        Create a preloader from a list of light curve identifiers.

        :param identifiers: The list of light curve identifiers.
        :param starting_index: The starting index to preload around.
        :return: The preloader.
        """
        preloader = cls()
        preloader.light_curve_identifiers = identifiers
        await preloader.load_light_curve_at_index_as_current(starting_index)
        return preloader