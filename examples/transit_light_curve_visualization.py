from pathlib import Path

from bokeh.io import show
from bokeh.plotting import figure as Figure

from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve


def main():
    light_curve_path = Path(
        'data/spoc_transit_experiment/train/positives/hlsp_tess-spoc_tess_phot_0000000004605846-s0044_tess_v1_lc.fits')
    light_curve = TessMissionLightCurve.from_path(light_curve_path)
    light_curve_figure = Figure(x_axis_label='Time (BTJD)', y_axis_label='Flux')
    light_curve_figure.circle(x=light_curve.times, y=light_curve.fluxes)
    light_curve_figure.line(x=light_curve.times, y=light_curve.fluxes, line_alpha=0.3)
    show(light_curve_figure)


if __name__ == '__main__':
    main()
