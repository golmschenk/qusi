"""
Code for vetting transit candidates.
"""
from ramjet.photometric_database.tess_target import TessTarget


class TransitVetter:
    """
    A class for vetting transit candidates.
    """
    @staticmethod
    def is_transit_depth_for_target_physical_for_planet(target: TessTarget, transit_depth: float) -> bool:
        """
        Check if the depth of a transit is not too deep to be caused by a planet for a given target.

        :param target: The target whose parameters should be used.
        :param transit_depth: The transit depth in relative flux.
        :return: A boolean stating if the depth is physical for a planet.
                 False meaning the radius of transiting body is (likely) too large to be a planet.
        """
        transiting_body_radius = target.calculate_transiting_body_radius(transit_depth)
        radius_of_jupiter__solar_radii = 0.10054
        planet_radius_threshold = 1.8 * radius_of_jupiter__solar_radii
        if transiting_body_radius < planet_radius_threshold:
            return True
        else:
            return False
