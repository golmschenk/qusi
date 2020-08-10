"""
Code to prepare the metadata tables.
"""
from ramjet.data_interface.tess_eclipsing_binary_metadata_manager import TessEclipsingBinaryMetadataManager
from ramjet.data_interface.tess_ffi_lightcurve_metadata_manager import TessFfiLightcurveMetadataManager
from ramjet.data_interface.tess_target_metadata_manager import TessTargetMetadataManger
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadataManager
from ramjet.data_interface.tess_two_minute_cadence_lightcurve_metadata_manager import \
    TessTwoMinuteCadenceLightcurveMetadataManger


def build_tables():
    """
    Prepare the metadata tables needed for TESS two minute transit related trials.
    """
    tess_ffi_lightcurve_metadata_manger = TessFfiLightcurveMetadataManager()
    tess_ffi_lightcurve_metadata_manger.build_table()
    tess_eclipsing_binary_metadata_manger = TessEclipsingBinaryMetadataManager()
    tess_eclipsing_binary_metadata_manger.build_table()
    tess_two_minute_cadence_lightcurve_metadata_manger = TessTwoMinuteCadenceLightcurveMetadataManger()
    tess_two_minute_cadence_lightcurve_metadata_manger.build_table()
    tess_transit_metadata_manager = TessTransitMetadataManager()
    tess_transit_metadata_manager.build_table()
    tess_target_metadata_manger = TessTargetMetadataManger()
    tess_target_metadata_manger.build_table()


if __name__ == '__main__':
    build_tables()
