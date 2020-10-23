"""
A model for the database entity of a TESS transiter.
"""
from peewee import AutoField, ForeignKeyField, FloatField, BooleanField

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.data_interface.tess_target_metadata_manager import TessTargetMetadata


class TessTransiter(MetadatabaseModel):
    """
    A database model for the database entity of a TESS transiter.
    """
    id = AutoField()
    target: TessTargetMetadata = ForeignKeyField(TessTargetMetadata)
    radius__solar_radii = FloatField(null=True)
    has_known_contamination_ratio = BooleanField(default=True)
    transit_epoch__btjd = FloatField()
    transit_period__days = FloatField()
    transit_duration__days = FloatField()
    transit_depth = FloatField()
