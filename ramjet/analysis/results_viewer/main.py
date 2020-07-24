import sys
from bokeh import plotting
from bokeh.io import curdoc

from ramjet.analysis.results_viewer.ffi_results_viewer import FfiResultsViewer
from ramjet.analysis.results_viewer.results_viewer import ResultsViewer


def main():
    plotting.output_notebook.__doc__ = ''
    document = curdoc()
    results_path = sys.argv[1]
    starting_index = int(sys.argv[2])
    ResultsViewer.attach_document(document, results_path, starting_index)


if __name__ == '__main__':
    main()
