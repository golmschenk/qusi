from bokeh.io import curdoc

from ramjet.analysis.results_viewer.results_viewer import ResultsViewer

document = curdoc()
results_path = '/Users/golmschenk/Desktop/SLB EN cont ac 2020-05-13-17-46-59 2020-05-14-15-05-31.feather'
ResultsViewer(document, results_path)
