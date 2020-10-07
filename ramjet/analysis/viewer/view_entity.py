"""Code grouping the objects related to one entity for the viewer."""

from ramjet.photometric_database.light_curve import LightCurve


class ViewEntity:
    """A class grouping the objects related to one entity for the viewer."""
    def __init__(self, index: int, light_curve: LightCurve):
        self.index = index
        self.light_curve = light_curve
