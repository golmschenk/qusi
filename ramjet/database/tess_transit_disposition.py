"""
A model for the database entity of a TESS transit disposition.
"""
from enum import Enum

from peewee import AutoField, ForeignKeyField, CharField, TextField

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.database.tess_transiter import TessTransiter


class Disposition(Enum):
    """
    An enum to represent the possible dispositions of a transiter.
    """
    GOOD = 'Good'
    CONDITIONAL = 'Conditional'
    MAYBE = 'Maybe'
    UNLIKELY = 'Unlikely'
    FAIL = 'Fail'


class Source(Enum):
    """
    An enum to represent the possible sources of dispositions of a transiter.
    """
    GREG_OLMSCHENK = 'Greg Olmschenk'
    GSFC_GROUP = 'GSFC group'


class TessTransitDisposition(MetadatabaseModel):
    """
    A database model for the database entity of a TESS transit disposition.
    """
    id = AutoField()
    transiter: TessTransiter = ForeignKeyField(TessTransiter)
    disposition = CharField(choices=Disposition)
    source = CharField(choices=Source)
    notes = TextField(null=True)
