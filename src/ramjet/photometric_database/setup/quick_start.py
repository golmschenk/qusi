"""
Code to download and prepare the data needed for the quick start tutorial.
"""
from pathlib import Path

from ramjet.data_interface.tess_data_interface import download_two_minute_cadence_light_curves
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.photometric_database.setup.tess_two_minute_cadence_transit_metadata import build_tables


def download_and_build_tables():
    """
    Downloads and prepares the data needed for the quick start tutorial.
    """
    download()
    build_tables()


def download():
    download_two_minute_cadence_light_curves(Path("data/tess_two_minute_cadence_light_curves"), limit=10000)
    tess_toi_data_interface = TessToiDataInterface()
    tess_toi_data_interface.download_exofop_toi_light_curves_to_directory(
        Path("data/tess_two_minute_cadence_light_curves")
    )


if __name__ == "__main__":
    download_and_build_tables()
