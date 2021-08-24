"""
A class for viewing FFI inference results.
"""
import numpy as np
from typing import Union

from ramjet.analysis.results_viewer.results_viewer import ResultsViewer
from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve, TessFfiColumnName

tess_data_interface = TessDataInterface()
tess_toi_data_interface = TessToiDataInterface()

class Target:
    def __init__(self, light_curve_path):
        self.loaded = False
        self.light_curve_path = light_curve_path
        self.tic_id, self.sector = TessFfiLightCurve.get_tic_id_and_sector_from_file_path(light_curve_path)
        self.pdcsap_fluxes: Union[np.ndarray, None] = None
        self.normalized_pdcsap_fluxes: Union[np.ndarray, None] = None
        self.pdcsap_flux_errors: Union[np.ndarray, None] = None
        self.normalized_pdcsap_flux_errors: Union[np.ndarray, None] = None
        self.sap_fluxes: Union[np.ndarray, None] = None
        self.normalized_sap_fluxes: Union[np.ndarray, None] = None
        self.load_light_curve()
        self.has_known_exofop_disposition = self.check_for_known_exofop_dispositions()
        tic_row = tess_data_interface.get_tess_input_catalog_row(self.tic_id)
        self.star_radius = tic_row['rad']
        self.loaded = True

    def load_light_curve(self):
        light_curve_path = self.light_curve_path
        self.pdcsap_fluxes, self.pdcsap_flux_errors, self.times = TessFfiLightCurve.load_fluxes_flux_errors_and_times_from_pickle_file(
            light_curve_path, TessFfiColumnName.CORRECTED_FLUX)
        nonnegative_pdcsap_fluxes = self.pdcsap_fluxes - np.minimum(np.nanmin(self.pdcsap_fluxes), 0)
        pdcsap_flux_median = np.nanmedian(nonnegative_pdcsap_fluxes)
        self.normalized_pdcsap_fluxes = nonnegative_pdcsap_fluxes / pdcsap_flux_median - 1
        self.normalized_pdcsap_flux_errors = self.pdcsap_flux_errors / pdcsap_flux_median
        self.sap_fluxes, _ = TessFfiLightCurve.load_fluxes_and_times_from_pickle_file(light_curve_path,
                                                                                      TessFfiColumnName.RAW_FLUX)
        nonnegative_sap_fluxes = self.sap_fluxes - np.minimum(np.nanmin(self.sap_fluxes), 0)
        sap_flux_median = np.nanmedian(nonnegative_sap_fluxes)
        self.normalized_sap_fluxes = nonnegative_sap_fluxes / sap_flux_median - 1

    def check_for_known_exofop_dispositions(self):
        dispositions = tess_toi_data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(self.tic_id)
        return dispositions.shape[0] > 0


class FfiResultsViewer(ResultsViewer):
    """
    A viewer inspecting FFI light curves of inference candidates.
    """
    def __init__(self, bokeh_document, results_path, starting_index: int = 0):
        super().__init__(bokeh_document, results_path, starting_index)
        self.target_type = Target
