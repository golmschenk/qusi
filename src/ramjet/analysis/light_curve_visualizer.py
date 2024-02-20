"""
Code for visualizing light curves.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import LinearColorMapper
from bokeh.palettes import Turbo256
from bokeh.plotting import figure as Figure
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from pathlib import Path


def plot_light_curve(
    times: np.ndarray,
    fluxes: np.ndarray,
    labels: np.ndarray = None,
    predictions: np.ndarray = None,
    title: str | None = None,
    x_label: str = "Days",
    y_label: str = "Flux",
    x_limits: (float, float) = (None, None),
    y_limits: (float, float) = (None, None),
    save_path: Path | str | None = None,
    *,
    exclude_flux_outliers: bool = False,
    base_data_point_size: float = 3,
):
    """
    Plots a light curve with a consistent styling. If true labels and/or predictions are included, these will
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
    with plt.style.context("seaborn-whitegrid"):
        figure, axes = plt.subplots(figsize=(16, 10))
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        color_map = plt.get_cmap("tab10")
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
        axes.scatter(
            times,
            fluxes,
            c=face_colors,
            marker="o",
            edgecolors=edge_colors,
            linewidths=base_data_point_size / 10,
            s=base_data_point_size,
            zorder=3,
        )
        if predictions is not None:
            axes.autoscale(False)
            transparent_prediction_color = (*prediction_color[:3], 0)
            prediction_color_map = LinearSegmentedColormap.from_list(
                "prediction-color-map", [transparent_prediction_color, prediction_color]
            )
            midpoints_between_times = (times[1:] + times[:-1]) / 2
            average_midpoint_distance = np.mean(np.diff(midpoints_between_times))
            extra_start_point = times[0] - average_midpoint_distance
            extra_end_point = times[-1] + average_midpoint_distance
            midpoints_between_times = np.concatenate([[extra_start_point], midpoints_between_times, [extra_end_point]])
            prediction_quad_mesh = axes.pcolormesh(
                midpoints_between_times, [0, 1], predictions[np.newaxis, :], cmap=prediction_color_map, vmin=0, vmax=1
            )
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
    if len(points.shape) != 1:
        msg = "Outlier removal only implemented for 1D data."
        raise ValueError(msg)
    median = np.nanmedian(points, axis=0)
    absolute_deviation_from_median = np.abs(points - median)
    median_absolute_deviation_from_median = np.nanmedian(absolute_deviation_from_median)
    modified_z_score = 0.6745 * absolute_deviation_from_median / median_absolute_deviation_from_median
    return modified_z_score > threshold


def create_dual_light_curve_figure(
    fluxes0, times0, name0, fluxes1, times1, name1, title, x_axis_label="Time (days)", y_axis_label="Relative flux"
) -> Figure:
    """
    Plots two light curves together. Mostly for comparing a light curve cleaned by two different methods.

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
    figure = Figure(title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label, active_drag="box_zoom")
    add_light_curve(figure, times0, fluxes0, name0, "firebrick")
    add_light_curve(figure, times1, fluxes1, name1, "mediumblue")
    return figure


def create_light_curve_figure(
    fluxes, times, name, title="", x_axis_label="Time (days)", y_axis_label="Relative flux"
) -> Figure:
    """
    Plots two light curves together. Mostly for comparing a light curve cleaned by two different methods.

    :param fluxes: The fluxes of the plot.
    :param times: The times of the plot.
    :param name: The name of the plot.
    :param title: The title of the figure.
    :param x_axis_label: The label of the x axis.
    :param y_axis_label: The label of the y axis.
    :return: The resulting figure.
    """
    figure = Figure(title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label, active_drag="box_zoom")
    add_light_curve(figure, times, fluxes, name, "mediumblue")
    return figure


def add_light_curve(figure, times, fluxes, legend_label, color):
    """Adds a light curve to the figure."""
    fluxes -= np.minimum(np.nanmin(fluxes), 0)
    flux_median = np.median(fluxes)
    figure.line(times, fluxes / flux_median, line_color=color, line_alpha=0.1)
    figure.circle(
        times,
        fluxes / flux_median,
        legend_label=legend_label,
        line_color=color,
        line_alpha=0.4,
        fill_color=color,
        fill_alpha=0.1,
    )


def add_folded_light_curve(figure, folded_times, fluxes, times):
    """Adds a light curve to the figure."""
    fluxes -= np.minimum(np.nanmin(fluxes), 0)
    flux_median = np.median(fluxes)
    relative_fluxes = fluxes / flux_median
    mapper = LinearColorMapper(palette=Turbo256, low=min(times), high=max(times))
    data_frame = pd.DataFrame({"folded_time": folded_times, "flux": relative_fluxes, "time": times})
    color = {"field": "time", "transform": mapper}
    figure.circle(
        source=data_frame, x="folded_time", y="flux", line_color=color, line_alpha=0.4, fill_color=color, fill_alpha=0.1
    )
    return figure


async def calculate_inlier_range(points: np.ndarray) -> (float, float):
    """
    Calculates the inlier range for a set of points.

    :param points: The points to get the range for.
    :return: The start and end of the inlier range.
    """
    outlier_indices = is_outlier(points)
    inliers = points[~outlier_indices]
    return float(np.nanmin(inliers)), float(np.nanmax(inliers))
