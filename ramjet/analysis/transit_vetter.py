"""
Code for vetting transit candidates.
"""
from ramjet.photometric_database.tess_target import TessTarget


class TransitVetter:
    """
    A class for vetting transit candidates.
    """
    @staticmethod
    def is_transit_depth_for_tic_id_physical_for_planet(transit_depth, tic_id):
        """
        Check if the depth of a transit is not too deep to be caused by a planet for a given target.

        :param transit_depth: The transit depth in relative flux.
        :param tic_id: The TIC ID of the target whose parameters should be used.
        :return: A boolean stating if the depth is physical for a planet.
                 False meaning the radius of transiting body is (likely) too large to be a planet.
        """
        target = TessTarget.from_tic_id(tic_id)
        transiting_body_radius = target.calculate_transiting_body_radius(transit_depth)
        radius_of_jupiter__solar_radii = 0.10054
        planet_radius_threshold = 1.8 * radius_of_jupiter__solar_radii
        if transiting_body_radius < planet_radius_threshold:
            return True
        else:
            return False
