"""Code grouping the objects related to one entity for the viewer."""
import asyncio
from typing import Any, Union

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.tess_target import TessTarget
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessTwoMinuteCadenceLightCurve


class ViewEntity:
    """A class grouping the objects related to one entity for the viewer."""
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.index: Union[int, None] = None
        self.confidence: Union[float, None] = None
        self.light_curve: Union[LightCurve, None] = None
        self.target: Union[TessTarget, None] = None

    @classmethod
    async def from_identifier_data_frame_row(cls, identifier_data_frame_row):
        """
        Creates the view entity based on a data frame row from an infer output.

        :param identifier_data_frame_row: The row of the data frame that should be used to prepare the view entity.
        :return: The view entity.
        """
        view_entity = cls()
        loop = asyncio.get_running_loop()
        light_curve_path_string = identifier_data_frame_row['light_curve_path']
        load_light_curve_task = loop.run_in_executor(None, view_entity.load_light_curve_from_identifier,
                                                     light_curve_path_string)
        tic_id, _ = cls.tess_data_interface.get_tic_id_and_sector_from_file_path(light_curve_path_string)
        load_target_task = loop.run_in_executor(None, TessTarget.from_tic_id, tic_id)
        light_curve, target = await asyncio.gather(load_light_curve_task, load_target_task)
        view_entity.index = identifier_data_frame_row['index']
        view_entity.confidence = identifier_data_frame_row['confidence']
        view_entity.light_curve = light_curve
        view_entity.target = target
        return view_entity

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
