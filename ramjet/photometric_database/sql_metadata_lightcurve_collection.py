"""
Code for a lightcurve collection that stores its metadata in the SQL database.
"""
import itertools
from pathlib import Path
from typing import Iterable
from peewee import Select

from ramjet.data_interface.metadatabase import MetadatabaseModel
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection, \
    LightcurveCollectionMethodNotImplementedError


class SqlMetadataLightcurveCollection(LightcurveCollection):
    """
    Class for a lightcurve collection that stores its metadata in the SQL database.
    """
    def __init__(self, repeat: bool = True):
        super().__init__()
        self.repeat: bool = repeat

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
        page_size = 1000
        query = self.get_sql_query()
        while True:
            for page_number in itertools.count(start=1, step=1):  # Peewee pages start on 1.
                page = query.paginate(page_number, paginate_by=page_size)
                if len(page) > 0:
                    for model in page:
                        yield Path(self.get_path_from_model(model))
                else:
                    break
            if not self.repeat:
                break