"""
Code to download and prepare the data needed for the quick start tutorial.
"""
from pathlib import Path

from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_target_metadata_manager import TessTargetMetadataManger
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadataManager
from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import \
    TessTwoMinuteCadenceLightcurveMetadataManger


def download():
    """
    Downloads and prepares the data needed for the quick start tutorial.
    """
    tess_data_interface = TessDataInterface()
    tess_data_interface.download_two_minute_cadence_lightcurves(
        Path('data/tess_two_minute_cadence_lightcurves'), limit=10000)
    tess_toi_data_interface = TessToiDataInterface()
    tess_toi_data_interface.download_exofop_toi_lightcurves_to_directory(
        Path('data/tess_two_minute_cadence_lightcurves'))
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()
    tess_two_minute_cadence_lightcurve_metadata_manger.build_table()
    tess_transit_metadata_manager = TessTransitMetadataManager()
    tess_transit_metadata_manager.build_table()
    tess_target_metadata_manger = TessTargetMetadataManger()
    tess_target_metadata_manger.build_table()


if __name__ == '__main__':
    download()
