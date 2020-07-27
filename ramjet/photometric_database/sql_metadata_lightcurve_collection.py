"""
Code for a lightcurve collection that stores its metadata in the SQL database.
"""
import itertools
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from peewee import Select, Field, Case

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection, \
    LightcurveCollectionMethodNotImplementedError


class SqlMetadataLightcurveCollection(LightcurveCollection):
    """
    Class for a lightcurve collection that stores its metadata in the SQL database.
    """
    def __init__(self):
        super().__init__()

    def get_sql_query(self) -> Select:
        """
        Gets the SQL query for the database models for the lightcurve collection.

        :return: The SQL query.
        """
        raise LightcurveCollectionMethodNotImplementedError

    def sql_count(self) -> int:
        """
        Gets the count of the rows returned by the SQL query.

        :return: The count.
        """
        return self.get_sql_query().count()

    def get_path_from_model(self, model: MetadatabaseModel) -> Path:
        """
        Gets the lightcurve path from the SQL database model.

        :return: The path to the lightcurve.
        """
        raise LightcurveCollectionMethodNotImplementedError

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the lightcurves in the collection.

        :return: An iterable of the lightcurve paths.
        """
        page_size = 10000
        query = self.get_sql_query()
        for page_number in itertools.count(start=1, step=1):  # Peewee pages start on 1.
            page = query.paginate(page_number, paginate_by=page_size)
            if len(page) > 0:
                for model in page:
                    yield Path(self.get_path_from_model(model))
            else:
                break

    @staticmethod
    def order_by_uuid_with_random_start(select_query: Select, uuid_field: Field) -> Select:
        """
        Applies an "order by" on a query using a passed UUID field. The "order by" starts at a random UUID and then
        loops back to the minimum UUID include all entities.

        :param select_query: The query to add the "order by" to.
        :param uuid_field: The UUID field to order on.
        :return: The query updated to include the "order by".
        """
        random_start_case = Case(None, [(uuid_field > uuid4(), 0)], 1)
        updated_select_query = select_query.order_by(random_start_case, uuid_field)
        return updated_select_query
