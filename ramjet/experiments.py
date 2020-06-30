"""
Code for common experiment component combinations (e.g., a database goes with a specific network model).
"""
from abc import ABC, abstractmethod
from tensorflow.keras import Model

from ramjet.models import SimpleLightcurveCnn
from ramjet.photometric_database.lightcurve_database import LightcurveDatabase
from ramjet.photometric_database.toi_database import ToiDatabase


class Experiment(ABC):
    """
    An abstract class representing the necessary components for an experiment to run.
    """
    def __init__(self):
        self.run_name: str
        self.database: LightcurveDatabase
        self.model: Model


class ToiExperiment(Experiment):
    """
    A class to represent the basic transit experiment.
    """
    def __init__(self):
        super().__init__()
        self.run_name = 'Transit'
        self.database = ToiDatabase()
        self.model = SimpleLightcurveCnn()
