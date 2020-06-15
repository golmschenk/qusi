"""
An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections.
"""
from abc import abstractmethod
from typing import List, Union

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection
from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class StandardAndInjectedLightcurveDatabase(LightcurveDatabase):
    """
    An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections
    to be used for training.
    """
    def __init__(self):
        super().__init__()
        self.standard_lightcurve_collections: List[LightcurveCollection] = []
        self.injectee_lightcurve_collection: Union[LightcurveCollection, None] = None
        self.injectable_lightcurve_collections: List[LightcurveCollection] = []
