"""
A class for viewing FFI inference results.
"""
import numpy as np
from typing import Union

from ramjet.analysis.results_viewer.results_viewer import ResultsViewer
from ramjet.data_interface.tess_data_interface import TessDataInterface, TessFluxType
from ramjet.data_interface.tess_ffi_data_interface import TessFfiDataInterface, FfiDataIndexes
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface

tess_data_interface = TessDataInterface()
tess_toi_data_interface = TessToiDataInterface()
tess_ffi_data_interface = TessFfiDataInterface()

class Target:
    def __init__(self, lightcurve_path):
        self.loaded = False
        self.lightcurve_path = lightcurve_path
        self.tic_id, self.sector = tess_ffi_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
        self.pdcsap_fluxes: Union[np.ndarray, None] = None
        self.normalized_pdcsap_fluxes: Union[np.ndarray, None] = None
        self.pdcsap_flux_errors: Union[np.ndarray, None] = None
        self.normalized_pdcsap_flux_errors: Union[np.ndarray, None] = None
        self.sap_fluxes: Union[np.ndarray, None] = None
        self.normalized_sap_fluxes: Union[np.ndarray, None] = None
        self.load_lightcurve()
        self.has_known_exofop_disposition = self.check_for_known_exofop_dispositions()
        tic_row = tess_data_interface.get_tess_input_catalog_row(self.tic_id)
        self.star_radius = tic_row['rad']
        self.loaded = True

    def load_lightcurve(self):
        lightcurve_path = self.lightcurve_path
        self.pdcsap_fluxes, self.pdcsap_flux_errors, self.times = tess_ffi_data_interface.load_fluxes_flux_errors_and_times_from_pickle_file(
            lightcurve_path, FfiDataIndexes.CORRECTED_FLUX)
        pdcsap_flux_median = np.nanmedian(self.pdcsap_fluxes)
        self.normalized_pdcsap_fluxes = self.pdcsap_fluxes / pdcsap_flux_median - 1
        self.normalized_pdcsap_flux_errors = self.pdcsap_flux_errors / pdcsap_flux_median
        self.sap_fluxes, _ = tess_ffi_data_interface.load_fluxes_and_times_from_pickle_file(lightcurve_path,
                                                                                            FfiDataIndexes.RAW_FLUX)
        self.normalized_sap_fluxes = self.sap_fluxes / np.nanmedian(self.sap_fluxes) - 1

    def check_for_known_exofop_dispositions(self):
        dispositions = tess_toi_data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(self.tic_id)
        return dispositions.shape[0] > 0


class FfiResultsViewer(ResultsViewer):
    """
    A viewer inspecting FFI lightcurves of inference candidates.
    """
    def __init__(self, bokeh_document, results_path):
        super().__init__(bokeh_document, results_path)
        self.target_type = Target
