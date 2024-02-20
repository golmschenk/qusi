"""
Code for a model for the TESS target database table.
"""

from peewee import AutoField, IntegerField

from ramjet.database.base_model import BaseModel


class TessTarget(BaseModel):
    """
    A model for the TESS target database table.
    """

    id = AutoField()  # noqa A003
    tic_id = IntegerField(index=True, unique=True)
