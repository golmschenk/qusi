import tempfile
import time
from pathlib import Path

import numpy as np
from astroquery.mast import Observations
from bokeh.io import show
from bokeh.plotting import figure as Figure

from ramjet.data_interface.tess_data_interface import download_products, \
    get_all_tess_spoc_light_curve_observations, get_product_list, download_spoc_light_curves_for_tic_ids, \
    get_spoc_target_data_frame_from_mast, download_spoc_light_curves_for_tic_ids_incremental
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve

# tic_ids = ['115419674']
tic_id = 115419674

spoc_target_data_frame = get_spoc_target_data_frame_from_mast()
spoc_target_tic_ids = spoc_target_data_frame['TIC_ID'].to_list()
light_curve_paths = download_spoc_light_curves_for_tic_ids_incremental(
    tic_ids=spoc_target_tic_ids, download_directory=Path('data/check_spoc'), limit=1000)
# TODO: Use shuffled SPOC target list to download targets up to limit specified.
for light_curve_path in light_curve_paths:
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
    print(median_time_diff)
    time.sleep(5)
    pass
