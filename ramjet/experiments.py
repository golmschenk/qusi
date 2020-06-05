"""
Code for common experiment component combinations (e.g., a database goes with a specific network model).
"""
from abc import ABC, abstractmethod
from tensorflow.keras import Model

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class Experiment(ABC):
    """
    An abstract class representing the necessary components for an experiment to run.
    """
    @property
    @abstractmethod
    def database(self) -> LightcurveDatabase:
        """
        :return: The database to use.
        """
        raise NotImplemented

    @property
    @abstractmethod
    def model(self) -> Model:
        """
        :return: The network model to use.
        """
        raise NotImplemented
