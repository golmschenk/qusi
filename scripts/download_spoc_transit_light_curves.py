from pathlib import Path

import numpy as np
from bokeh.io import show
from bokeh.plotting import figure as Figure

from ramjet.data_interface.tess_data_interface import \
    get_spoc_tic_id_list_from_mast, download_spoc_light_curves_for_tic_ids_incremental
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface, ToiColumns
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve

negative_paths = Path('scripts/data/spoc_transit_experiment/negatives')
# spoc_target_tic_ids = get_spoc_tic_id_list_from_mast()
# negative_light_curve_paths = download_spoc_light_curves_for_tic_ids_incremental(
#     tic_ids=spoc_target_tic_ids, download_directory=Path('data/spoc_transit_experiment/negatives'), limit=3000)
# tess_toi_data_interface = TessToiDataInterface()
# suspected_planet_tic_ids = tess_toi_data_interface.toi_dispositions[
#     tess_toi_data_interface.toi_dispositions[ToiColumns.disposition.value] != 'FP'][ToiColumns.tic_id.value]
# positive_light_curve_paths = download_spoc_light_curves_for_tic_ids_incremental(
#     tic_ids=spoc_target_tic_ids, download_directory=Path('data/spoc_transit_experiment/positives'), limit=1000)
sectors = []
median_time_diffs = []
negative_light_curve_paths = negative_paths.glob('*.fits')
for light_curve_path in negative_light_curve_paths:
    light_curve = TessMissionLightCurve.from_path(light_curve_path)
    times = light_curve.times[~np.isnan(light_curve.times)]
    median_time_diff = np.nanmedian(np.diff(times))
    sector = light_curve.sector
    sectors.append(sector)
    median_time_diffs.append(median_time_diff)
figure = Figure()
figure.circle(x=sectors, y=median_time_diffs)
show(figure)
