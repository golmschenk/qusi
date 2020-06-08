"""
An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections.
"""
from abc import abstractmethod

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class StandardAndInjectedLightcurveDatabase(LightcurveDatabase):
    """
    An abstract class allowing for any number and combination of standard and injectable/injectee lightcurve collections.
    """
    pass
