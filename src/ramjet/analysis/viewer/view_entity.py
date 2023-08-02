"""Code grouping the objects related to one entity for the viewer."""
import asyncio
from pathlib import Path
from typing import Any, Union

from ramjet.analysis.transit_vetter import TransitVetter
from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve
from ramjet.photometric_database.tess_light_curve import TessLightCurve
from ramjet.photometric_database.tess_target import TessTarget


class ViewEntity:
    """A class grouping the objects related to one entity for the viewer."""
    tess_data_interface = TessDataInterface()
    tess_toi_data_interface = TessToiDataInterface()
    vetter = TransitVetter()

    def __init__(self):
        self.index: Union[int, None] = None
        self.confidence: Union[float, None] = None
        self.light_curve: Union[TessLightCurve, None] = None
        self.target: Union[TessTarget, None] = None
        self.has_exofop_dispositions: Union[bool, None] = None
        self.has_nearby_toi: Union[bool, None] = None

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
        light_curve_path = Path(light_curve_path_string)
        load_light_curve_task = loop.run_in_executor(None, view_entity.load_light_curve_from_identifier,
                                                     light_curve_path)
        tic_id, sector = TessFfiLightCurve.get_tic_id_and_sector_from_file_path(light_curve_path)
        load_has_dispositions_task = loop.run_in_executor(
            None, view_entity.tess_toi_data_interface.has_any_exofop_dispositions_for_tic_id, tic_id
        )
        tic_id_only_target = TessTarget()  # Create a stand-in target to use for other parallel preloading.
        tic_id_only_target.tic_id = tic_id
        load_nearby_toi_task = loop.run_in_executor(
            None, view_entity.vetter.has_nearby_toi_targets, tic_id_only_target
        )
        load_target_task = loop.run_in_executor(None, TessTarget.from_tic_id, tic_id)
        light_curve, target, has_dispositions, has_nearby_toi = await asyncio.gather(
            load_light_curve_task, load_target_task, load_has_dispositions_task, load_nearby_toi_task)
        view_entity.has_nearby_toi = has_nearby_toi
        view_entity.has_exofop_dispositions = has_dispositions
        view_entity.index = identifier_data_frame_row['index']
        view_entity.confidence = identifier_data_frame_row['confidence']
        view_entity.light_curve = light_curve
        view_entity.target = target
        return view_entity

    @staticmethod
    def load_light_curve_from_identifier(identifier: Any) -> TessLightCurve:
        """
        Loads a light curve from a generic identifier.

        :param identifier: The identifier of the light curve.
        :return: The light curve.
        """
        light_curve = TessFfiLightCurve.from_path(identifier)
        light_curve.convert_to_relative_scale()
        return light_curve
