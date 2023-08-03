from pathlib import Path

import numpy as np
from astroquery.mast import Observations
from bokeh.io import show
from bokeh.plotting import figure as Figure

from ramjet.data_interface.tess_data_interface import download_products
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessTwoMinuteCadenceLightCurve

tic_ids = ['7903477', '7583207', '115419674']
obsTable = Observations.query_criteria(provenance_name="TESS-SPOC",
                                       target_name=tic_ids)

data = Observations.get_product_list(obsTable)
obs_data_frame = obsTable.to_pandas()
data_data_frame = data.to_pandas()

download_lc = download_products(data_data_frame, 'data/check_spoc')
downloaded_light_curves_data_frame = download_lc
light_curve_path = Path(downloaded_light_curves_data_frame['Local Path'].iloc[6])
light_curve = TessTwoMinuteCadenceLightCurve.from_path(light_curve_path)
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
