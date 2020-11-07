"""
Code for a model for the TESS target database table.
"""
from typing import List

from peewee import IntegerField, AutoField

from ramjet.database.base_model import BaseModel
from ramjet.database.tess_planet_disposition import TessPlanetDisposition, Disposition
from ramjet.database.tess_transiter import TessTransiter


class TessTarget(BaseModel):
    """
    A model for the TESS target database table.
    """
    id = AutoField()
    tic_id = IntegerField(index=True, unique=True)

    @staticmethod
    def get_tic_ids_of_passing_vetted_transiting_planet_candidates() -> List[int]:
        """
        Gets the TIC IDs of candidates which have passed vetting of having planet transits.

        :return: The list of candidate TIC IDs.
        """
        candidate_tic_id_query = TessTarget.select(TessTarget.tic_id).join(TessTransiter).where(
            TessTransiter.id.in_(
                TessPlanetDisposition.select(TessPlanetDisposition.transiter).where(
                    TessPlanetDisposition.disposition == Disposition.PASS.value))
            &
            TessTransiter.id.not_in(
                TessPlanetDisposition.select(TessPlanetDisposition.transiter).where(
                    TessPlanetDisposition.disposition == Disposition.FAIL.value))
        ).order_by(TessTarget.tic_id)
        return list(candidate_tic_id_query)
