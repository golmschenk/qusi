"""
Code to represent a TESS target.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import requests
from astroquery.gaia import Gaia

from ramjet.data_interface.tess_data_interface import get_tess_input_catalog_row


class TessTarget:
    """
    A class to represent an TESS target. Usually a star or star system.
    """

    def __init__(self):
        self.tic_id: int | None = None
        self.radius: float | None = None
        self.mass: float | None = None
        self.magnitude: float | None = None
        self.contamination_ratio: float | None = None

    @classmethod
    def from_tic_id(cls, tic_id: int) -> TessTarget:
        """
        Creates a target from a TIC ID.

        :param tic_id: The TIC ID to create the target from.
        :return: The target.
        """
        target = TessTarget()
        target.tic_id = tic_id
        tic_row = get_tess_input_catalog_row(target.tic_id)
        target.radius = tic_row["rad"]
        if np.isnan(target.radius):
            gaia_source_id_string = tic_row["GAIA"]
            if pd.notna(gaia_source_id_string):
                target.radius = target.get_radius_from_gaia(int(gaia_source_id_string))
        target.mass = tic_row["mass"]
        # noinspection SpellCheckingInspection
        target.magnitude = tic_row["Tmag"]
        # noinspection SpellCheckingInspection
        target.contamination_ratio = tic_row["contratio"]
        return target

    @staticmethod
    def get_radius_from_gaia(gaia_source_id: int) -> float:
        """
        Retrieves the radius of a body from Gaia.

        :param gaia_source_id: The Gaia source ID for the target to retrieve the radius of.
        :return: The radius.
        """
        # noinspection SqlResolve
        gaia_job = Gaia.launch_job(f"select * from gaiadr2.gaia_source where source_id={gaia_source_id}")
        query_results_data_frame = gaia_job.get_results().to_pandas()
        radius = query_results_data_frame["radius_val"].iloc[0]
        return radius

    def calculate_transiting_body_radius(
        self, transit_depth: float, *, allow_unknown_contamination_ratio: bool = False
    ) -> float:
        """
        Calculates the radius of a transiting body based on the target parameters and the transit depth.

        :param transit_depth: The depth of the transit signal.
        :param allow_unknown_contamination_ratio: Whether to calculate even without a known contamination ratio.
        :return: The calculated radius of the transiting body.
        """
        contamination_ratio = self.contamination_ratio
        if pd.isna(contamination_ratio):
            if allow_unknown_contamination_ratio:
                contamination_ratio = 0
            else:
                error_message = f"Contamination ratio {contamination_ratio} cannot be used to calculate the radius."
                raise ValueError(error_message)
        return self.radius * math.sqrt(transit_depth * (1 + contamination_ratio))

    def retrieve_nearby_tic_targets(self) -> pd.DataFrame:
        """
        Retrieves the data frame of nearby targets from ExoFOP.

        :return: The data frame of nearby targets.
        """
        csv_url = f"https://exofop.ipac.caltech.edu/tess/download_nearbytarget.php?id={self.tic_id}&output=csv"
        csv_string = requests.get(csv_url, timeout=600).content.decode("utf-8")
        if "Distance Err" not in csv_string:  # Correct ExoFOP bug where distance error column header is missing.
            csv_string = csv_string.replace("Distance(pc)", "Distance (pc),Distance Err (pc)")
        data_frame = pd.read_csv(io.StringIO(csv_string), index_col=False)
        data_frame = data_frame[data_frame["TIC ID"] != self.tic_id]  # Remove row of current target.
        return data_frame
