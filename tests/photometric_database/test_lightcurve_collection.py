import numpy as np
from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


class TestLightcurveCollection:
    def test_lightcurve_collection_has_necessary_attributes(self):
        lightcurve_collection = LightcurveCollection(
            function_to_get_lightcurve_paths=lambda: [],
            function_to_load_times_and_fluxes_from_lightcurve_path=lambda path: (np.array([]), np.array([])),
            label=0)
        assert hasattr(lightcurve_collection, 'get_lightcurve_paths')
        assert hasattr(lightcurve_collection, 'load_times_and_fluxes_from_lightcurve_path')
        assert hasattr(lightcurve_collection, 'label')
