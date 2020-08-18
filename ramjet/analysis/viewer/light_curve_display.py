"""
Code for displaying a light curve figure.
"""
from typing import Union, List

from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure


class LightCurveDisplay:
    """
    A class for displaying a light curve figure.
    """
    def __init__(self):
        self.figure: Union[Figure, None] = None
        self.data_source: Union[ColumnDataSource, None] = None
        self.time_column_name: Union[str, None] = None
        self.flux_column_names: List[str] = []
