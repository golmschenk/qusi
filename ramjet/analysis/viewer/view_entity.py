from typing import NamedTuple

from ramjet.photometric_database.light_curve import LightCurve


class ViewEntity(NamedTuple):
    """A simple tuple linking an index and a light curve."""
    index: int
    light_curve: LightCurve