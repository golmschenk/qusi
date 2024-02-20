import sys
from pathlib import Path

from bokeh import plotting
from bokeh.io import curdoc

from ramjet.analysis.light_curve_folding_vizualizer.viewer import Viewer
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve


def main():
    plotting.output_notebook.__doc__ = ""
    document = curdoc()
    light_curve_path = Path(sys.argv[1])
    light_curve = TessFfiLightCurve.from_path(light_curve_path)
    Viewer(document, light_curve, title=light_curve_path.name)


if __name__ == "__main__" or "bokeh_app_" in __name__:
    main()
