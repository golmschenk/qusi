"""
Code for a model for the TESS target database table.
"""
from typing import List

from peewee import IntegerField, AutoField

from ramjet.database.base_model import BaseModel


class TessTarget(BaseModel):
    """
    A model for the TESS target database table.
    """
    id = AutoField()
    tic_id = IntegerField(index=True, unique=True)
