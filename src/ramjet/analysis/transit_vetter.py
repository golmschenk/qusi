"""
Code for vetting transit candidates.
"""
import pandas as pd

from ramjet.photometric_database.tess_target import TessTarget


class TransitVetter:
    """
    A class for vetting transit candidates.
    """

    radius_of_jupiter__solar_radii = 0.1028

    def is_transit_depth_for_target_physical_for_planet(self, target: TessTarget, transit_depth: float) -> bool:
        """
        Check if the depth of a transit is not too deep to be caused by a planet for a given target.

        :param target: The target whose parameters should be used.
        :param transit_depth: The transit depth in relative flux.
        :return: A boolean stating if the depth is physical for a planet.
                 False meaning the radius of transiting body is (likely) too large to be a planet.
        """
        transiting_body_radius = target.calculate_transiting_body_radius(transit_depth)
        planet_radius_threshold = 1.8 * self.radius_of_jupiter__solar_radii
        return transiting_body_radius < planet_radius_threshold

    @staticmethod
    def has_no_nearby_likely_eclipsing_binary_background_targets(target: TessTarget) -> bool:
        """
        Checks if the target has likely nearby targets which may be eclipsing binaries showing up as the transit.

        :param target: The target of interest.
        :return: Whether or not there is at least one problematic nearby target.
        """
        nearby_threshold_arcseconds = 21  # 21 arcseconds is the size of the side of a TESS pixel.
        magnitude_difference_threshold = 5
        nearby_target_data_frame = target.retrieve_nearby_tic_targets()
        problematic_nearby_target_data_frame = nearby_target_data_frame.loc[
            (nearby_target_data_frame["TESS Mag"] < target.magnitude + magnitude_difference_threshold)
            & (nearby_target_data_frame["Separation (arcsec)"] < nearby_threshold_arcseconds)
        ]
        return problematic_nearby_target_data_frame.shape[0] == 0

    @staticmethod
    def has_nearby_toi_targets(target: TessTarget) -> bool:
        """
        Checks if the target has nearby TOI targets which are likely to be where the transit is from.

        :param target: The target of interest.
        :return: Whether or not there is at least one problematic nearby target.
        """
        nearby_threshold_arcseconds = 31.5  # 1.5 TESS pixels.
        nearby_target_data_frame = target.retrieve_nearby_tic_targets()
        problematic_nearby_target_data_frame = nearby_target_data_frame.loc[
            (pd.notnull(nearby_target_data_frame["TOI"]))
            & (nearby_target_data_frame["Separation (arcsec)"] < nearby_threshold_arcseconds)
        ]
        return problematic_nearby_target_data_frame.shape[0] != 0

    def get_maximum_physical_depth_for_planet_for_target(
        self, target: TessTarget, *, allow_missing_contamination_ratio: bool = False
    ) -> float:
        """
        Determines the maximum depth allowable for a given target for a transit to be caused by a planet.

        :param target: The target to check for.
        :param allow_missing_contamination_ratio: Allow for unknown contamination, which will then default to 0.
        :return: The maximum relative depth allowed.
        """
        maximum_planet_radius = 1.8 * self.radius_of_jupiter__solar_radii
        contamination_ratio = target.contamination_ratio
        if pd.isna(contamination_ratio):
            if allow_missing_contamination_ratio:
                contamination_ratio = 0
            else:
                error_message = f"Contamination ratio {contamination_ratio} is not a number."
                raise ValueError(error_message)
        maximum_physical_depth = (maximum_planet_radius**2) / ((target.radius**2) * (1 + contamination_ratio))
        return maximum_physical_depth
