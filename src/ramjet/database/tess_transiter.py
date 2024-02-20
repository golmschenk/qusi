"""
A model for the database entity of a TESS transiter.
"""
from peewee import AutoField, BooleanField, FloatField, ForeignKeyField

from ramjet.database.base_model import BaseModel
from ramjet.database.tess_target import TessTarget


class TessTransiter(BaseModel):
    """
    A database model for the database entity of a TESS transiter.
    """

    id = AutoField()  # noqa A003
    target: TessTarget = ForeignKeyField(TessTarget)
    radius__solar_radii = FloatField(null=True)
    has_known_contamination_ratio = BooleanField(default=True)
    transit_epoch__btjd = FloatField()
    transit_period__days = FloatField()
    transit_duration__days = FloatField()
    transit_relative_depth = FloatField()
