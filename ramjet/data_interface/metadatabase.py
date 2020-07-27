"""
Code for the the metadatabase.
"""
from peewee import Model, SqliteDatabase

metadatabase = SqliteDatabase('data/metadatabase.sqlite3',
                              pragmas={'journal_mode': 'wal'})


def convert_class_to_table_name(model_class: Model):
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
