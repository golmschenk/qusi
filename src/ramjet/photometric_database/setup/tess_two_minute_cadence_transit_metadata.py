"""
Code to prepare the metadata needed for TESS two minute transit related trials.
"""
from ramjet.data_interface.tess_target_metadata_manager import TessTargetMetadataManger
from ramjet.data_interface.tess_transit_metadata_manager import TessTransitMetadataManager
from ramjet.data_interface.tess_two_minute_cadence_light_curve_metadata_manager import (
    TessTwoMinuteCadenceLightCurveMetadataManger,
)


def build_tables():
    """
    Prepare the metadata tables needed for TESS two minute transit related trials.
    """
    tess_two_minute_cadence_light_curve_metadata_manger = TessTwoMinuteCadenceLightCurveMetadataManger()
    tess_two_minute_cadence_light_curve_metadata_manger.build_table()
    tess_transit_metadata_manager = TessTransitMetadataManager()
    tess_transit_metadata_manager.build_table()
    tess_target_metadata_manger = TessTargetMetadataManger()
    tess_target_metadata_manger.build_table()


if __name__ == "__main__":
    build_tables()
