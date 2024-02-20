"""
Code for the the base model of the database to build other models from.
"""

from peewee import Model, SqliteDatabase

database = SqliteDatabase("data/database.sqlite3", pragmas={"journal_mode": "wal"}, check_same_thread=False)


def convert_class_to_table_name(model_class: type[Model]):
    """
    Creates the table name based on the model class.

    :param model_class: The class to create the table name for.
    :return: The name of the table.
    """
    model_name = model_class.__name__
    return model_name


class BaseModel(Model):
    """
    A general model for the database tables.
    """

    class Meta:
        """The meta information for the database models."""

        database = database
        table_function = convert_class_to_table_name
