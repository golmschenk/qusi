import sys
from pathlib import Path

from bokeh import plotting
from bokeh.io import curdoc

from ramjet.analysis.light_curve_folding_vizualizer.viewer import Viewer
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve


def main():
    plotting.output_notebook.__doc__ = ''
    document = curdoc()
    # results_path = sys.argv[1]
    # starting_index = int(sys.argv[2])
    light_curve_path = Path(
        'logs/go/gathered/data/tess_ffi_light_curves/tesslcs_sector_6_104/tesslcs_tmag_12_13/tesslc_71970184.pkl')
    light_curve = TessFfiLightCurve.from_path(light_curve_path)
    Viewer(document, light_curve)


if __name__ == '__main__' or 'bokeh_app_' in __name__:
    main()
