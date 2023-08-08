import tempfile
from pathlib import Path

import numpy as np
from astroquery.mast import Observations
from bokeh.io import show
from bokeh.plotting import figure as Figure

from ramjet.data_interface.tess_data_interface import download_products, \
    get_all_tess_spoc_light_curve_observations, get_product_list
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve

tic_ids = ['115419674']
tic_id = 115419674

obsTable = Observations.query_criteria(provenance_name="TESS-SPOC",
                                       target_name=tic_ids)
tess_observations = Observations.query_criteria(obs_collection='HLSP', dataproduct_type='timeseries',
                                                provenance_name='TESS-SPOC',
                                                calib_level=4,  # Science data product level.
                                                target_name=tic_id)
# data = Observations.get_product_list(obsTable)
obs_data_frame = obsTable.to_pandas()
tess_observations_data_frame = tess_observations.to_pandas()
# data_data_frame = data.to_pandas()
light_curve_observations = get_all_tess_spoc_light_curve_observations(tic_id=tic_id)
light_curve_data_product_list = get_product_list(light_curve_observations)
light_curve_data_products = download_products(light_curve_data_product_list, data_directory=Path(tempfile.gettempdir()))

download_lc = download_products(light_curve_observations, 'data/check_spoc')
downloaded_light_curves_data_frame = download_lc
light_curve_path = Path(downloaded_light_curves_data_frame['Local Path'].iloc[6])
light_curve = TessMissionLightCurve.from_path(light_curve_path)
for column_name in light_curve.data_frame:
    light_curve.data_frame[column_name] = light_curve.data_frame[column_name].values.byteswap().newbyteorder()
light_curve.data_frame = light_curve.data_frame[light_curve.data_frame['pdcsap_flux'].notna()]
figure = Figure()
figure.circle(x=light_curve.times, y=light_curve.fluxes)
figure.line(x=light_curve.times, y=light_curve.fluxes, line_alpha=0.2)
show(figure)
times = light_curve.times
median_time_diff = np.median(np.diff(light_curve.times))
pass
