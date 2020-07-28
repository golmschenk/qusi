"""
Code for the the metadatabase.
"""
import uuid
from typing import Type

from peewee import Model, SqliteDatabase

metadatabase = SqliteDatabase('data/metadatabase.sqlite3',
                              pragmas={'journal_mode': 'wal'})
metadatabase_uuid_namespace = uuid.UUID('ed5c78c4-d8dd-4525-9633-97beac696cd1')


def convert_class_to_table_name(model_class: Type[Model]):
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


def metadatabase_uuid(name: str):
    """
    Generates a reproducible UUID for the metadatabase based on a name string.

    :param name: The string used to produce the UUID.
    :return: The UUID
    """
    uuid.uuid5(metadatabase_uuid_namespace, name)
