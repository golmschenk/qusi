"""
Code to download and prepare the data needed for the GPU experiments.
"""
from pathlib import Path

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.photometric_database.setup.tess_two_minute_cadence_transit_metadata import build_tables


def download():
    """
    Downloads and prepares the data needed for the GPU experiments.
    """
    tess_data_interface = TessDataInterface()
    tess_data_interface.download_two_minute_cadence_lightcurves(Path('data/tess_two_minute_cadence_lightcurves'))
    tess_toi_data_interface = TessToiDataInterface()
    tess_toi_data_interface.download_exofop_toi_lightcurves_to_directory(
        Path('data/tess_two_minute_cadence_lightcurves'))
    build_tables()


if __name__ == '__main__':
    download()
