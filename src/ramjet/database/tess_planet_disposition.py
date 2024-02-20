"""
A model for the database entity of a TESS planet disposition.
"""
from enum import Enum

from peewee import AutoField, CharField, ForeignKeyField, TextField

from ramjet.database.base_model import BaseModel
from ramjet.database.tess_target import TessTarget
from ramjet.database.tess_transiter import TessTransiter


class Disposition(Enum):
    """
    An enum to represent the possible planet dispositions.
    """

    PASS = "Pass"  # noqa S105 : False positive assuming field is a password field.
    CONDITIONAL = "Conditional"
    AMBIGUOUS = "Ambiguous"
    UNLIKELY = "Unlikely"
    FAIL = "Fail"
    REPROCESSING_REQUIRED = "Reprocessing required"
    KNOWN = "Known"


class Source(Enum):
    """
    An enum to represent the possible sources of planet dispositions.
    """

    GREG_OLMSCHENK = "Greg Olmschenk"
    GSFC_GROUP = "GSFC group"


class TessPlanetDisposition(BaseModel):
    """
    A database model for the database entity of a TESS planet disposition.
    """

    id = AutoField()  # noqa A003
    transiter: TessTransiter = ForeignKeyField(TessTransiter)
    disposition = CharField(choices=Disposition)
    source = CharField(choices=Source)
    notes = TextField(null=True)

    class Meta:
        """Schema meta data for the model."""

        indexes = ((("source", "transiter"), True),)

    @staticmethod
    def get_tic_ids_of_passing_vetted_transiting_planet_candidates() -> list[int]:
        """
        Gets the TIC IDs of candidates which have passed vetting of having planet transits.

        :return: The list of candidate TIC IDs.
        """
        candidate_tic_id_query = (
            TessTarget.select(TessTarget.tic_id)
            .join(TessTransiter)
            .where(
                TessTransiter.id.in_(
                    TessPlanetDisposition.select(TessPlanetDisposition.transiter).where(
                        TessPlanetDisposition.disposition == Disposition.PASS.value
                    )
                )
                & TessTransiter.id.not_in(
                    TessPlanetDisposition.select(TessPlanetDisposition.transiter).where(
                        TessPlanetDisposition.disposition == Disposition.FAIL.value
                    )
                )
            )
            .order_by(TessTarget.tic_id)
        )
        return [candidate.tic_id for candidate in candidate_tic_id_query]
