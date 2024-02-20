from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import pandas as pd
import requests

from ramjet.data_interface.tess_data_interface import (
    download_products,
    get_all_two_minute_single_sector_observations,
    get_product_list,
)

logger = logging.getLogger(__name__)


class ToiColumns(Enum):
    """
    An enum for the names of the TOI columns for Pandas data frames.
    """

    tic_id = "TIC ID"
    disposition = "Disposition"
    transit_epoch__bjd = "Transit epoch (BJD)"
    transit_period__days = "Transit period (days)"
    transit_duration = "Transit duration (hours)"
    sector = "Sector"


class ExofopDisposition(Enum):
    """
    An enum for the ExoFOP dispositions.
    """

    CONFIRMED_PLANET = "CP"
    KEPLER_CONFIRMED_PLANET = "KP"
    PLANET_CANDIDATE = "PC"
    FALSE_POSITIVE = "FP"


class TessToiDataInterface:
    """
    A data interface for working with the TESS table of objects of interest.
    """

    def __init__(self, data_directory="data/tess_toi"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.toi_dispositions_path = self.data_directory.joinpath("toi_dispositions.csv")
        self.ctoi_dispositions_path = self.data_directory.joinpath("ctoi_dispositions.csv")
        self.light_curves_directory = self.data_directory.joinpath("light_curves")
        self.toi_dispositions_: pd.DataFrame | None = None
        self.ctoi_dispositions_: pd.DataFrame | None = None

    @property
    def toi_dispositions(self):
        """
        The TOI dispositions data frame property. Will load as an instance attribute on first access. Updates from
        ExoFOP on first access.

        :return: The TOI dispositions data frame.
        """
        if self.toi_dispositions_ is None:
            try:
                self.update_toi_dispositions_file()
            except requests.exceptions.ConnectionError:
                logger.warning("Unable to connect to update TOI file. Attempting to use existing file...")
            self.toi_dispositions_ = self.load_toi_dispositions_in_project_format()
        return self.toi_dispositions_

    def update_toi_dispositions_file(self):
        """
        Downloads the latest TOI dispositions file.
        """
        toi_csv_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        response = requests.get(toi_csv_url, timeout=600)
        with self.toi_dispositions_path.open("wb") as csv_file:
            csv_file.write(response.content)

    @property
    def ctoi_dispositions(self):
        """
        The CTOI dispositions data frame property. Will load as an instance attribute on first access. Updates from
        ExoFOP on first access.

        :return: The CTOI dispositions data frame.
        """
        if self.ctoi_dispositions_ is None:
            try:
                self.update_ctoi_dispositions_file()
            except requests.exceptions.ConnectionError:
                logger.warning("Unable to connect to update TOI file. Attempting to use existing file...")
            self.ctoi_dispositions_ = self.load_ctoi_dispositions_in_project_format()
        return self.ctoi_dispositions_

    def update_ctoi_dispositions_file(self):
        """
        Downloads the latest CTOI dispositions file.
        """
        ctoi_csv_url = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
        response = requests.get(ctoi_csv_url, timeout=600)
        with self.ctoi_dispositions_path.open("wb") as csv_file:
            csv_file.write(response.content)

    def load_toi_dispositions_in_project_format(self) -> pd.DataFrame:
        """
        Loads the ExoFOP TOI table information from CSV to a data frame using a project consistent naming scheme.

        :return: The data frame of the TOI dispositions table.
        """
        columns_to_use = ["TIC ID", "TFOPWG Disposition", "Epoch (BJD)", "Period (days)", "Duration (hours)", "Sectors"]
        dispositions = pd.read_csv(self.toi_dispositions_path, usecols=columns_to_use)
        dispositions.rename(
            columns={
                "TFOPWG Disposition": ToiColumns.disposition.value,
                "Epoch (BJD)": ToiColumns.transit_epoch__bjd.value,
                "Period (days)": ToiColumns.transit_period__days.value,
                "Duration (hours)": ToiColumns.transit_duration.value,
                "Sectors": ToiColumns.sector.value,
            },
            inplace=True,
        )
        dispositions[ToiColumns.disposition.value] = dispositions[ToiColumns.disposition.value].fillna("")
        dispositions = dispositions[dispositions[ToiColumns.sector.value].notna()]
        dispositions[ToiColumns.sector.value] = dispositions[ToiColumns.sector.value].str.split(",")
        dispositions = dispositions.explode(ToiColumns.sector.value)
        dispositions[ToiColumns.sector.value] = pd.to_numeric(dispositions[ToiColumns.sector.value]).astype(
            pd.Int64Dtype()
        )
        return dispositions

    def load_ctoi_dispositions_in_project_format(self) -> pd.DataFrame:
        """
        Loads the ExoFOP CTOI table information from CSV to a data frame using a project consistent naming scheme.

        :return: The data frame of the CTOI dispositions table.
        """
        columns_to_use = ["TIC ID", "TFOPWG Disposition", "Transit Epoch (BJD)", "Period (days)", "Duration (hrs)"]
        dispositions = pd.read_csv(self.ctoi_dispositions_path, usecols=columns_to_use)
        dispositions.rename(
            columns={
                "TFOPWG Disposition": ToiColumns.disposition.value,
                "Transit Epoch (BJD)": ToiColumns.transit_epoch__bjd.value,
                "Period (days)": ToiColumns.transit_period__days.value,
                "Duration (hrs)": ToiColumns.transit_duration.value,
            },
            inplace=True,
        )
        dispositions[ToiColumns.disposition.value] = dispositions[ToiColumns.disposition.value].fillna("")
        return dispositions

    def retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(self, tic_id: int) -> pd.DataFrame:
        """
        Retrieves the ExoFOP disposition information for a given TIC ID from <https://exofop.ipac.caltech.edu/tess/>`_.

        :param tic_id: The TIC ID to get available data for.
        :return: The disposition data frame.
        """
        toi_dispositions = self.toi_dispositions
        ctoi_dispositions = self.ctoi_dispositions
        toi_and_coi_dispositions = pd.concat([toi_dispositions, ctoi_dispositions], axis=0, ignore_index=True)
        tic_target_dispositions = toi_and_coi_dispositions[toi_and_coi_dispositions["TIC ID"] == tic_id]
        return tic_target_dispositions

    def has_any_exofop_dispositions_for_tic_id(self, tic_id: int) -> bool:
        """
        Returns whether or not any dispositions exist for this TIC ID.

        :param tic_id: The TIC ID to check.
        :return: True if there are dispositions, False if none.
        """
        existing_dispositions = self.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(tic_id)
        return existing_dispositions.shape[0] != 0

    def print_exofop_toi_and_ctoi_planet_dispositions_for_tic_target(self, tic_id):
        """
        Prints all ExoFOP disposition information for a given TESS target.

        :param tic_id: The TIC target to for.
        """
        dispositions_data_frame = self.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(tic_id)
        if dispositions_data_frame.shape[0] == 0:
            logger.info("No known ExoFOP dispositions found.")
            return
        # Use context options to not truncate printed data.
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
            logger.info(dispositions_data_frame)

    def download_exofop_toi_light_curves_to_directory(self, directory: Path):
        """
        Downloads the `ExoFOP database <https://exofop.ipac.caltech.edu/tess/view_toi.php>`_ light curve files to the
        given directory.

        :param directory: The directory to download the light curves to. Defaults to the data interface directory.
        """
        logger.info("Downloading ExoFOP TOI disposition CSV...")
        if isinstance(directory, str):
            directory = Path(directory)
        tic_ids = self.toi_dispositions[ToiColumns.tic_id.value].unique()
        logger.info("Downloading TESS observation list...")
        single_sector_observations = get_all_two_minute_single_sector_observations(tic_ids)
        logger.info("Downloading light curves which are confirmed or suspected planets in TOI dispositions...")
        suspected_planet_dispositions = self.toi_dispositions[
            self.toi_dispositions[ToiColumns.disposition.value] != "FP"
        ]
        suspected_planet_observations = pd.merge(
            single_sector_observations,
            suspected_planet_dispositions,
            how="inner",
            on=[ToiColumns.tic_id.value, ToiColumns.sector.value],
        )
        suspected_planet_data_products = get_product_list(suspected_planet_observations)
        suspected_planet_light_curve_data_products = suspected_planet_data_products[
            suspected_planet_data_products["productFilename"].str.endswith("lc.fits")
        ]
        suspected_planet_download_manifest = download_products(
            suspected_planet_light_curve_data_products, data_directory=self.data_directory
        )
        logger.info(f"Verifying and moving light curves to {directory}...")
        directory.mkdir(parents=True, exist_ok=True)
        for _row_index, row in suspected_planet_download_manifest.iterrows():
            if row["Status"] == "COMPLETE":
                file_path = Path(row["Local Path"])
                file_path.rename(directory.joinpath(file_path.name))


if __name__ == "__main__":
    tess_toi_data_interface = TessToiDataInterface()
    tess_toi_data_interface.download_exofop_toi_light_curves_to_directory(
        Path("data/tess_two_minute_cadence_light_curves")
    )
