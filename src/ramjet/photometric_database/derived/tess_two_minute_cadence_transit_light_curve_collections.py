"""
Code representing the collection of TESS two minute cadence light curves containing transits.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ramjet.data_interface.tess_transit_metadata_manager import Disposition, TessTransitMetadata
from ramjet.data_interface.tess_two_minute_cadence_light_curve_metadata_manager import (
    TessTwoMinuteCadenceLightCurveMetadata,
)
from ramjet.photometric_database.derived.tess_two_minute_cadence_light_curve_collection import (
    TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection,
)

if TYPE_CHECKING:
    from peewee import Select


class TessTwoMinuteCadenceConfirmedTransitLightCurveCollection(
    TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection
):
    """
    A class representing the collection of TESS two minute cadence light curves containing transits.
    """

    def __init__(self, dataset_splits: list[int] | None = None):
        super().__init__(dataset_splits=dataset_splits)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            TessTransitMetadata.disposition == Disposition.CONFIRMED.value
        )
        query = query.where(TessTwoMinuteCadenceLightCurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessTwoMinuteCadenceConfirmedAndCandidateTransitLightCurveCollection(
    TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection
):
    """
    A class representing the collection of TESS two minute cadence light curves containing transits.
    """

    def __init__(self, dataset_splits: list[int] | None = None):
        super().__init__(dataset_splits=dataset_splits)
        self.label = 1

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
            | (TessTransitMetadata.disposition == Disposition.CANDIDATE.value)
        )
        query = query.where(TessTwoMinuteCadenceLightCurveMetadata.tic_id.in_(transit_tic_id_query))
        return query


class TessTwoMinuteCadenceNonTransitLightCurveCollection(TessTwoMinuteCadenceTargetDatasetSplitLightCurveCollection):
    """
    A class representing the collection of TESS two minute cadence light curves containing transits.
    """

    def __init__(self, dataset_splits: list[int] | None = None):
        super().__init__(dataset_splits=dataset_splits)
        self.label = 0

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the light curve collection.

        :return: The SQL query.
        """
        query = super().get_sql_query()
        transit_candidate_tic_id_query = TessTransitMetadata.select(TessTransitMetadata.tic_id).where(
            (TessTransitMetadata.disposition == Disposition.CONFIRMED.value)
            | (TessTransitMetadata.disposition == Disposition.CANDIDATE.value)
        )
        query = query.where(TessTwoMinuteCadenceLightCurveMetadata.tic_id.not_in(transit_candidate_tic_id_query))
        return query
