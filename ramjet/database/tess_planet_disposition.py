"""
A model for the database entity of a TESS planet disposition.
"""
from enum import Enum

from peewee import ForeignKeyField, CharField, TextField, AutoField

from ramjet.database.base_model import BaseModel
from ramjet.database.tess_transiter import TessTransiter


class Disposition(Enum):
    """
    An enum to represent the possible planet dispositions.
    """
    PASS = 'Pass'
    CONDITIONAL = 'Conditional'
    AMBIGUOUS = 'Ambiguous'
    UNLIKELY = 'Unlikely'
    FAIL = 'Fail'
    REPROCESSING_REQUIRED = 'Reprocessing required'
    KNOWN = 'Known'


class Source(Enum):
    """
    An enum to represent the possible sources of planet dispositions.
    """
    GREG_OLMSCHENK = 'Greg Olmschenk'
    GSFC_GROUP = 'GSFC group'


class TessPlanetDisposition(BaseModel):
    """
    A database model for the database entity of a TESS planet disposition.
    """
    id = AutoField()
    transiter: TessTransiter = ForeignKeyField(TessTransiter)
    disposition = CharField(choices=Disposition)
    source = CharField(choices=Source)
    notes = TextField(null=True)

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('source', 'transiter'), True),
        )
