"""
Code to download and prepare the data needed for the GPU experiments.
"""
from pathlib import Path

from ramjet.data_interface.tess_data_interface import download_two_minute_cadence_light_curves
from ramjet.photometric_database.setup.tess_two_minute_cadence_transit_metadata import build_tables


def download():
    """
    Downloads and prepares the data needed for the GPU experiments.
    """
    download_two_minute_cadence_light_curves(Path("data/tess_two_minute_cadence_light_curves"))
    build_tables()


if __name__ == "__main__":
    download()
