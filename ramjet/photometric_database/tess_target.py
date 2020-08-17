"""
Code to represent a TESS target.
"""
from __future__ import annotations

from typing import Union

from ramjet.data_interface.tess_data_interface import TessDataInterface


class TessTarget:
    """
    A class to represent an TESS target. Usually a star or star system.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self):
        self.tic_id: Union[float, None] = None
        self.radius: Union[float, None] = None
        self.mass: Union[float, None] = None
        self.magnitude: Union[float, None] = None

    @classmethod
    def from_tic_id(cls, tic_id: int) -> TessTarget:
        """
        Creates a target from a TIC ID.

        :param tic_id: The TIC ID to create the target from.
        :return: The target.
        """
        target = TessTarget()
        target.tic_id = tic_id
        tic_row = cls.tess_data_interface.get_tess_input_catalog_row(target.tic_id)
        target.radius = tic_row['rad']
        target.mass = tic_row['mass']
        target.magnitude = tic_row['Tmag']
        return target
