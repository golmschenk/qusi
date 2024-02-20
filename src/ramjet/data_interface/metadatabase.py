"""
Code for the the metadatabase.
"""
from uuid import UUID, uuid5

from peewee import Model, SqliteDatabase

metadatabase = SqliteDatabase("data/metadatabase.sqlite3", pragmas={"journal_mode": "wal"}, check_same_thread=False)
metadatabase_uuid_namespace = UUID("ed5c78c4-d8dd-4525-9633-97beac696cd1")


def convert_class_to_table_name(model_class: type[Model]):
    """
    Creates the table name based on the model class.

    :param model_class: The class to create the table name for.
    :return: The name of the table.
    """
    model_name = model_class.__name__
    return model_name


class MetadatabaseModel(Model):
    """
    A general model for the metadatabase tables.
    """

    class Meta:
        """The meta information for the metadatabase models."""

        database = metadatabase
        table_function = convert_class_to_table_name
        primary_key = False


def metadatabase_uuid(name: str) -> UUID:
    """
    Generates a reproducible UUID for the metadatabase based on a name string.

    :param name: The string used to produce the UUID.
    :return: The UUID.
    """
    return uuid5(metadatabase_uuid_namespace, name)


def dataset_split_from_uuid(uuid: UUID) -> int:
    """
    Generates a repeatable dataset split from a UUID.

    :param uuid: The UUID to seed with.
    :return: The dataset split.
    """
    return uuid.int % 10
