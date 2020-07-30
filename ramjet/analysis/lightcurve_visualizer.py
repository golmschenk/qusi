"""
Code for visualizing lightcurves.
"""

from pathlib import Path
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from bokeh.colors import Color
from bokeh.layouts import gridplot
from bokeh.models import Column
from bokeh.plotting import Figure
from matplotlib.colors import LinearSegmentedColormap

from ramjet.analysis.color_palette import lightcurve_color0, lightcurve_color1, lightcurve_color3


def plot_lightcurve(times: np.ndarray, fluxes: np.ndarray, labels: np.ndarray = None, predictions: np.ndarray = None,
                    title: str = None, x_label: str = 'Days', y_label: str = 'Flux',
                    x_limits: (float, float) = (None, None), y_limits: (float, float) = (None, None),
                    save_path: Union[Path, str] = None, exclude_flux_outliers: bool = False,
                    base_data_point_size: float = 3):
    """
    Plots a lightcurve with a consistent styling. If true labels and/or predictions are included, these will
    additionally be plotted.

    :param times: The times of the measurements.
    :param fluxes: The fluxes of the measurements.
    :param labels: The binary labels for each time step.
    :param predictions: The probability prediction for each time step.
    :param title: The title of the plot.
    :param x_label: The label for the x axis.
    :param y_label: The label for the y axis.
    :param x_limits: Optional axis limiting for the x axis.
    :param y_limits: Optional axis limiting for the y axis.
    :param save_path: The path to save the plot to. If `None`, the plot will be shown instead.
    :param exclude_flux_outliers: Whether or not to exclude flux outlier data points when plotting.
    :param base_data_point_size: The size of the data points to use when plotting (and related sizes).
    """
    with plt.style.context('seaborn-whitegrid'):
        figure, axes = plt.subplots(figsize=(16, 10))
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        color_map = plt.get_cmap('tab10')
        data_point_color = color_map(0)
        positive_data_point_color = color_map(2)
        prediction_color = color_map(3)
        if exclude_flux_outliers:
            outlier_indices = is_outlier(fluxes)
            fluxes = fluxes[~outlier_indices]
            times = times[~outlier_indices]
            if labels is not None:
                labels = labels[~outlier_indices]
            if predictions is not None:
                predictions = predictions[~outlier_indices]
        if labels is not None:
            edge_colors = np.where(labels.reshape(-1, 1), [positive_data_point_color], [data_point_color])
            face_colors = np.copy(edge_colors)
            face_colors[:, 3] = 0.2
        else:
            edge_colors = [data_point_color]
            face_colors = [(*data_point_color[:3], 0.2)]
        axes.scatter(times, fluxes, c=face_colors, marker='o', edgecolors=edge_colors,
                     linewidths=base_data_point_size / 10, s=base_data_point_size, zorder=3)
        if predictions is not None:
            axes.autoscale(False)
            transparent_prediction_color = (*prediction_color[:3], 0)
            prediction_color_map = LinearSegmentedColormap.from_list('prediction-color-map',
                                                                     [transparent_prediction_color, prediction_color])
            midpoints_between_times = (times[1:] + times[:-1]) / 2
            average_midpoint_distance = np.mean(np.diff(midpoints_between_times))
            extra_start_point = times[0] - average_midpoint_distance
            extra_end_point = times[-1] + average_midpoint_distance
            midpoints_between_times = np.concatenate([[extra_start_point], midpoints_between_times, [extra_end_point]])
            prediction_quad_mesh = axes.pcolormesh(midpoints_between_times, [0, 1], predictions[np.newaxis, :],
                                                   cmap=prediction_color_map, vmin=0, vmax=1)
            transformation = axes.get_xaxis_transform()
            prediction_quad_mesh.set_transform(transformation)
            axes.grid(True)  # Re-enable the grid since pcolormesh disables it.
        if title is not None:
            axes.set_title(title)
        figure.patch.set_alpha(0)  # Transparent figure background while keeping grid background.
        # Need to explicitly state face color or the save will override it.
        axes.set_xlim(x_limits[0], x_limits[1])
        axes.set_ylim(y_limits[0], y_limits[1])
        if save_path is not None:
            plt.savefig(save_path, facecolor=figure.get_facecolor(), dpi=400)
        else:
            plt.show()
        plt.close(figure)


def is_outlier(points: np.ndarray, threshold: float = 5):
    """
    Uses the median absolute deviation to determine if the input data points are "outliers" for the purpose of
    plotting.

    :param points: The observations to search for outliers in.
    :param threshold: The modified z-score to use as a threshold. Observations with a modified z-score based on the
                      median absolute deviation greater than this value will be classified as outliers.
    """
    assert len(points.shape) == 1  # Only designed to work with 1D data.
    median = np.median(points, axis=0)
    absolute_deviation_from_median = np.abs(points - median)
    median_absolute_deviation_from_median = np.median(absolute_deviation_from_median)
    modified_z_score = 0.6745 * absolute_deviation_from_median / median_absolute_deviation_from_median
    return modified_z_score > threshold


def create_dual_lightcurve_figure(fluxes0, times0, name0, fluxes1, times1, name1, title, x_axis_label='Time (days)',
                                  y_axis_label='Relative flux') -> Figure:
    """
    Plots two lightcurves together. Mostly for comparing a lightcurve cleaned by two different methods.

    :param fluxes0: The fluxes of the first plot.
    :param times0: The times of the first plot.
    :param name0: The name of the first plot.
    :param fluxes1: The fluxes of the second plot.
    :param times1: The times of the second plot.
    :param name1: The name of the second plot.
    :param title: The title of the figure.
    :param x_axis_label: The label of the x axis.
    :param y_axis_label: The label of the y axis.
    :return: The resulting figure.
    """
    figure = Figure(title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label, active_drag='box_zoom')

    def add_lightcurve(times, fluxes, legend_label, color):
        """Adds a lightcurve to the figure."""
        fluxes -= np.minimum(np.nanmin(fluxes), 0)
        flux_median = np.median(fluxes)
        figure.line(times, fluxes / flux_median, line_color=color, line_alpha=0.1)
        figure.circle(times, fluxes / flux_median, legend_label=legend_label, line_color=color, line_alpha=0.4,
                      fill_color=color, fill_alpha=0.1)
    add_lightcurve(times0, fluxes0, name0, 'firebrick')
    add_lightcurve(times1, fluxes1, name1, 'mediumblue')
    return figure


def add_lightcurve_plot_to_figure(figure: Figure, times: np.ndarray, fluxes: np.ndarray, color: Color,
                                  legend_label: Union[str, None] = None):
    """
    Adds a lightcurve plot to a figure.

    :param figure: The figure to add the lightcurve to.
    :param times: The times of the lightcurve.
    :param fluxes: The fluxes of the lightcurve.
    :param color: The color to use for plotting.
    :param legend_label: The label to give the lightcurve in the legend of the figure.
    """
    figure.line(times, fluxes, line_color=color, line_alpha=0.1)
    kwargs = {}
    if legend_label is not None:
        kwargs['legend_label'] = legend_label  # Bokeh handles None differently than a none existent kwarg.
    figure.circle(times, fluxes, line_color=color, line_alpha=0.4, fill_color=color, fill_alpha=0.1, **kwargs)


def create_plotted_lightcurve_figure(times: np.ndarray, fluxes: np.ndarray, title: Union[str, None] = None,
                                     x_axis_label: Union[str, None] = 'Times',
                                     y_axis_label: Union[str, None] = 'Fluxes') -> Figure:
    """
    Create a lightcurve figure and plot a lightcurve to it.

    :param times: The times of the lightcurve.
    :param fluxes: The fluxes of the lightcurve.
    :param title: The title of the figure.
    :param x_axis_label: The label for the x axis.
    :param y_axis_label: The label for the y axis.
    :return: The figure with plotted lightcurve.
    """
    figure = Figure(title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label, active_drag='box_zoom')
    add_lightcurve_plot_to_figure(figure, times, fluxes, lightcurve_color0)
    figure.sizing_mode = 'stretch_width'
    return figure


def create_plotted_injected_lightcurve_components_column(
        injectee_times: np.ndarray, injectee_fluxes: np.ndarray, injectable_times: np.ndarray,
        injectable_magnitudes: np.ndarray, injected_times: np.ndarray, injected_fluxes: np.ndarray) -> Column:
    """
    Creates a Bokeh column containing the two plots displaying the injectee and injected lightcurves along side the
    injectable signal.

    :param injectee_times: The times of the lightcurve which will have a signal injected into it.
    :param injectee_fluxes: The fluxes of the lightcurve which will have a signal injected into it.
    :param injectable_times: The times of the signal to be injected.
    :param injectable_magnitudes: The magnitudes of the signal to be injected.
    :param injected_times: The times of the lightcurve with injected signal.
    :param injected_fluxes: The fluxes of the lightcurve with injected signal.
    :return: The column containing the figures.
    """
    injectee_and_injected_figure = Figure(x_axis_label='Time', y_axis_label='Fluxes', active_drag='box_zoom')
    injectable_figure = Figure(x_axis_label='Time', y_axis_label='Fluxes', active_drag='box_zoom',
                               x_range=injectee_and_injected_figure.x_range)

    add_lightcurve_plot_to_figure(injectee_and_injected_figure, injectee_times, injectee_fluxes, lightcurve_color0,
                                  legend_label='Injectee')
    add_lightcurve_plot_to_figure(injectee_and_injected_figure, injected_times, injected_fluxes, lightcurve_color1,
                                  'Injected')
    add_lightcurve_plot_to_figure(injectable_figure, injectable_times, injectable_magnitudes, lightcurve_color3,
                                  'Injectable')
    grid_plot = gridplot([[injectee_and_injected_figure], [injectable_figure]])
    return grid_plot
