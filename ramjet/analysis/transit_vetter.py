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

    @staticmethod
    def has_problematic_nearby_targets(target: TessTarget) -> bool:
        """
        Checks if the target has problematic nearby targets.

        :param target: The target of interest.
        :return: Whether or not there is at least one problematic nearby target.
        """
        nearby_threshold_arcseconds = 21  # 21 arcseconds is the size of the side of a TESS pixel.
        magnitude_difference_threshold = 5
        nearby_target_data_frame = target.retrieve_nearby_tic_targets()
        problematic_nearby_target_data_frame = nearby_target_data_frame.loc[
            (nearby_target_data_frame['TESS Mag'] < target.magnitude + magnitude_difference_threshold) &
            (nearby_target_data_frame['Separation (arcsec)'] < nearby_threshold_arcseconds)
        ]
        if problematic_nearby_target_data_frame.shape[0] == 0:
            return True
        else:
            return False
