"""
Code to lightcurves predicted by Ramjet in 2 minute data.
"""
import datetime
import math
import time
from concurrent.futures import ThreadPoolExecutor, Future
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Union, Dict, List

from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.command.util import build_single_handler_application
from bokeh.document import without_document_lock
from bokeh.models import ColumnDataSource, LinearColorMapper, Column, Button, Row, Div, Range1d, DataRange1d, CustomJS, \
    TableColumn, DataTable
from bokeh.events import Tap
from bokeh.plotting import Figure
from bokeh.server.server import Server

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
from tornado import gen

from ramjet.analysis.lightcurve_visualizer import is_outlier
from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
from ramjet.data_interface.tess_data_interface import TessDataInterface, TessFluxType

tess_data_interface = TessDataInterface()
tess_toi_data_interface = TessToiDataInterface()


class Target:
    def __init__(self, lightcurve_path):
        self.loaded = False
        self.tic_id, self.sector = tess_data_interface.get_tic_id_and_sector_from_file_path(lightcurve_path)
        self.pdcsap_fluxes: Union[np.ndarray, None] = None
        self.normalized_pdcsap_fluxes: Union[np.ndarray, None] = None
        self.pdcsap_flux_errors: Union[np.ndarray, None] = None
        self.normalized_pdcsap_flux_errors: Union[np.ndarray, None] = None
        self.sap_fluxes: Union[np.ndarray, None] = None
        self.normalized_sap_fluxes: Union[np.ndarray, None] = None
        self.load_lightcurve()
        self.has_known_exofop_disposition = self.check_for_known_exofop_dispositions()
        tic_row = tess_data_interface.get_tess_input_catalog_row(self.tic_id)
        self.star_radius = tic_row['rad']
        self.loaded = True

    def load_lightcurve(self):
        lightcurve_path = tess_data_interface.download_lightcurve(self.tic_id, self.sector)
        self.pdcsap_fluxes, self.pdcsap_flux_errors, self.times = tess_data_interface.load_fluxes_flux_errors_and_times_from_fits_file(
            lightcurve_path, TessFluxType.PDCSAP, remove_nans=False)
        nonnegative_pdcsap_fluxes = self.pdcsap_fluxes - np.minimum(np.nanmin(self.pdcsap_fluxes), 0)
        pdcsap_flux_median = np.nanmedian(nonnegative_pdcsap_fluxes)
        self.normalized_pdcsap_fluxes = nonnegative_pdcsap_fluxes / pdcsap_flux_median - 1
        self.normalized_pdcsap_flux_errors = self.pdcsap_flux_errors / pdcsap_flux_median
        self.sap_fluxes, _ = tess_data_interface.load_fluxes_and_times_from_fits_file(lightcurve_path,
                                                                                      TessFluxType.SAP,
                                                                                      remove_nans=False)
        nonnegative_sap_fluxes = self.sap_fluxes - np.minimum(np.nanmin(self.sap_fluxes), 0)
        sap_flux_median = np.nanmedian(nonnegative_sap_fluxes)
        self.normalized_sap_fluxes = nonnegative_sap_fluxes / sap_flux_median - 1

    def check_for_known_exofop_dispositions(self):
        dispositions = tess_toi_data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(self.tic_id)
        return dispositions.shape[0] > 0


class ResultsViewer:
    def __init__(self, bokeh_document, results_path, starting_index: int = 0):
        self.bokeh_document = bokeh_document
        self.target_type = Target
        self.pool_executor = ThreadPoolExecutor()
        self.results_data_frame = pd.read_csv(results_path, index_col='Index')
        self.current_target_index = starting_index
        self.number_of_indexes_before_and_after_to_load = 2
        self.number_of_indexes_before_and_after_to_delete = 5
        self.target_future_dictionary: Dict[Future] = {}
        self.target_dictionary: Dict[Future] = {}
        self.previous_button, self.next_button = self.create_target_switching_buttons()
        self.target_title_div = Div()
        self.target_title_div.sizing_mode = 'stretch_width'
        self.lightcurve_figure, self.lightcurve_data_source = self.create_flux_comparison_lightcurve_figure()
        self.known_planet_div = Div(text='This target has a known ExoFOP disposition',
                                    css_classes=['notification', 'is-warning', 'is-hidden', 'is-fullwidth'])

        self.fold_coordinate_data_source = ColumnDataSource({'Time (BTJD)': [], 'Normalized PDCSAP flux': []})
        self.event_coordinates = []
        unfolded_lightcurve_figure = self.create_unfolded_lightcurve_figure(self.lightcurve_data_source)
        folded_lightcurve_figure = self.create_folded_figured_based_on_clicks_in_unfolded_figure(unfolded_lightcurve_figure,
                                                                                                 self.lightcurve_data_source)
        self.star_radius = None
        self.depth = None
        self.period = None
        self.transit_epoch = None
        self.initial_fit_data_source = ColumnDataSource({'Folded time (days)': [], 'Relative flux': [],
                                                         'Fit': [], 'Fit time': [], 'Time (BTJD)': [],
                                                         'Time (days)': []})
        self.parameters_table_data_source = ColumnDataSource(pd.DataFrame())
        run_fitting_button = Button(label='Run fitting')
        initial_fit_figure, parameters_table = self.create_mcmc_fit_figures(run_fitting_button)
        self.add_to_negatives_button = self.create_add_to_negatives_button()

        self.add_to_negatives_button.name = 'add_to_negatives_button'
        bokeh_document.add_root(self.add_to_negatives_button)
        self.previous_button.name = 'previous_button'
        bokeh_document.add_root(self.previous_button)
        self.next_button.name = 'next_button'
        bokeh_document.add_root(self.next_button)
        self.target_title_div.name = 'target_title_div'
        bokeh_document.add_root(self.target_title_div)
        self.lightcurve_figure.name = 'lightcurve_figure'
        unfolded_lightcurve_figure.name = 'unfolded_lightcurve_figure'
        folded_lightcurve_figure.name = 'folded_lightcurve_figure'
        initial_fit_figure.name = 'initial_fit_figure'
        js_reset = CustomJS(code='''
            Bokeh.documents[0].get_model_by_name("unfolded_lightcurve_figure").reset.emit()
            Bokeh.documents[0].get_model_by_name("folded_lightcurve_figure").reset.emit()
            Bokeh.documents[0].get_model_by_name("initial_fit_figure").reset.emit()
        ''')
        self.lightcurve_data_source.js_on_change('data', js_reset)  # Hacky workaround until the Bokeh team makes a better solution. https://github.com/bokeh/bokeh/issues/9218
        bokeh_document.add_root(self.lightcurve_figure)
        self.known_planet_div.name = 'known_planet_div'
        bokeh_document.add_root(self.known_planet_div)
        bokeh_document.add_root(unfolded_lightcurve_figure)
        bokeh_document.add_root(folded_lightcurve_figure)
        run_fitting_button.name = 'run_fitting_button'
        parameters_table.name = 'parameters_table'
        bokeh_document.add_root(run_fitting_button)
        bokeh_document.add_root(initial_fit_figure)
        bokeh_document.add_root(parameters_table)


    @classmethod
    def attach_document(cls, bokeh_document, results_path, starting_index: int = 0):
        viewer = cls(bokeh_document, results_path, starting_index)
        viewer.load_surrounding_lightcurves()
        viewer.display_current_target()
        return viewer

    def create_mcmc_fit_figures(self, run_fitting_button):
        self_ = self
        initial_fit_figure = Figure(x_axis_label='Folded time (days)', y_axis_label='Relative flux',
                                    title=f'Initial fit')
        parameters_table_columns = [TableColumn(field=column, title=column) for column in
                                    ['parameter', 'mean', 'sd', 'r_hat']]
        parameters_table = DataTable(source=self.parameters_table_data_source, columns=parameters_table_columns,
                                     editable=True)
        initial_fit_data_source = self.initial_fit_data_source
        bokeh_document = self.bokeh_document

        @gen.coroutine
        def update_initial_fit_figure(fluxes, gp_pred, inds, lc_pred, map_soln, relative_times, times, x_fold):
            initial_fit_data_source.data['Time (BTJD)'] = times
            initial_fit_data_source.data['Time (days)'] = relative_times
            initial_fit_data_source.data['Folded time (days)'] = x_fold
            initial_fit_data_source.data['Relative flux'] = fluxes - gp_pred - map_soln["mean"]
            initial_fit_data_source.data['Fit'] = lc_pred[inds] - map_soln["mean"]
            initial_fit_data_source.data['Fit time'] = x_fold[
                inds]  # TODO: This is terrible, you should be able to line them up *afterward* to not make a duplicate time column

        @gen.coroutine
        @without_document_lock
        def fit(self, map_soln, model):
            with model:
                trace = pm.sample(
                    tune=2000,
                    draws=2000,
                    start=map_soln,
                    chains=4,
                    step=xo.get_dense_nuts_step(target_accept=0.9),
                )
            trace_summary = pm.summary(trace, round_to='none')  # Not a typo. PyMC3 wants 'none' as a string here.
            epoch = round(trace_summary['mean']['Transit epoch (BTJD)'], 3)  # Round the epoch differently, as BTJD needs more digits.
            trace_summary['mean'] = self_.round_series_to_significant_figures(trace_summary['mean'], 5)
            trace_summary['mean']['Transit epoch (BTJD)'] = epoch
            self.bokeh_document.add_next_tick_callback(partial(self.update_parameters_table, trace_summary))
            with pd.option_context('display.max_columns', None, 'display.max_rows', None):
                print(trace_summary)
                print(f'Star radius: {self.star_radius}')
            # TODO: This should not happen automatically. Only after a button click.
            # scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            # credentials = Credentials.from_service_account_file(
            #     'ramjet/analysis/google_spreadsheet_credentials.json', scopes=scopes)
            # gc = gspread.authorize(credentials)
            # sh = gc.open('Ramjet transit candidates shared for vetting')
            # worksheet = sh.get_worksheet(0)
            # # Find first empty row.
            # empty_row_index = 1
            # for row_index in itertools.count(start=1):
            #     row_values = worksheet.row_values(row_index)
            #     if len(row_values) == 0:
            #         empty_row_index = row_index
            #         break
            # worksheet.update_cell(empty_row_index, 1, self_.tic_id)
            # worksheet.update_cell(empty_row_index, 2, str(self_.sectors).replace('[', '').replace(']', '')),
            # worksheet.update_cell(empty_row_index, 3, trace_summary['mean']['Transit epoch (BTJD)'])
            # worksheet.update_cell(empty_row_index, 4, trace_summary['mean']['period'])
            # worksheet.update_cell(empty_row_index, 5, trace_summary['mean']['Transit depth (relative flux)'])
            # worksheet.update_cell(empty_row_index, 6, trace_summary['mean']['Transit duration (days)'])
            # worksheet.update_cell(empty_row_index, 7, self_.star_radius)
            # worksheet.update_cell(empty_row_index, 8, trace_summary['mean']['Planet radius (solar radii)'] * self_.star_radius)

        @gen.coroutine
        @without_document_lock
        def run_fitting():
            times = self.lightcurve_data_source.data['Time (BTJD)'].astype(np.float32)
            flux_errors = self.lightcurve_data_source.data['Normalized PDCSAP flux error']
            fluxes = self.lightcurve_data_source.data['Normalized PDCSAP flux']
            relative_times = self.lightcurve_data_source.data['Time (days)']
            nan_indexes = np.union1d(np.argwhere(np.isnan(fluxes)), np.union1d(np.argwhere(np.isnan(times)),
                                                                               np.argwhere(np.isnan(flux_errors))))
            fluxes = np.delete(fluxes, nan_indexes)
            flux_errors = np.delete(flux_errors, nan_indexes)
            times = np.delete(times, nan_indexes)
            relative_times = np.delete(relative_times, nan_indexes)
            with pm.Model() as model:
                # Stellar parameters
                mean = pm.Normal("mean", mu=0.0, sigma=10.0 * 1e-3)
                u = xo.distributions.QuadLimbDark("u")
                star_params = [mean, u]

                # Gaussian process noise model
                sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.nanmedian(flux_errors))
                log_Sw4 = pm.Normal("log_Sw4", mu=0.0, sigma=10.0)
                log_w0 = pm.Normal("log_w0", mu=np.log(2 * np.pi / 10.0), sigma=10.0)
                kernel = xo.gp.terms.SHOTerm(log_Sw4=log_Sw4, log_w0=log_w0, Q=1.0 / 3)
                noise_params = [sigma, log_Sw4, log_w0]

                # Planet parameters
                log_ror = pm.Normal("log_ror", mu=0.5 * np.log(self_.depth), sigma=10.0 * 1e-3)
                ror = pm.Deterministic("ror", tt.exp(log_ror))
                depth = pm.Deterministic('Transit depth (relative flux)', tt.square(ror))
                planet_radius = pm.Deterministic('Planet radius (solar radii)', ror * self_.star_radius)

                # Orbital parameters
                log_period = pm.Normal("log_period", mu=np.log(self_.period), sigma=1.0)
                t0 = pm.Normal('Transit epoch (BTJD)', mu=self_.transit_epoch, sigma=1.0)
                log_dur = pm.Normal("log_dur", mu=np.log(0.1), sigma=10.0)
                b = xo.distributions.ImpactParameter("b", ror=ror)

                period = pm.Deterministic('Transit period (days)', tt.exp(log_period))
                dur = pm.Deterministic('Transit duration (days)', tt.exp(log_dur))

                # Set up the orbit
                orbit = xo.orbits.KeplerianOrbit(period=period, duration=dur, t0=t0, b=b, r_star=self.star_radius)

                # We're going to track the implied density for reasons that will become clear later
                pm.Deterministic("rho_circ", orbit.rho_star)

                # Set up the mean transit model
                star = xo.LimbDarkLightCurve(u)

                def lc_model(t):
                    return mean + tt.sum(
                        star.get_light_curve(orbit=orbit, r=ror * self.star_radius, t=t), axis=-1
                    )

                # Finally the GP observation model
                gp = xo.gp.GP(kernel, times, (flux_errors ** 2) + (sigma ** 2), mean=lc_model)
                gp.marginal("obs", observed=fluxes)

                # Double check that everything looks good - we shouldn't see any NaNs!
                print(model.check_test_point())

                # Optimize the model
                map_soln = model.test_point
                map_soln = xo.optimize(map_soln, [sigma])
                map_soln = xo.optimize(map_soln, [log_ror, b, log_dur])
                map_soln = xo.optimize(map_soln, noise_params)
                map_soln = xo.optimize(map_soln, star_params)
                map_soln = xo.optimize(map_soln)

            with model:
                gp_pred, lc_pred = xo.eval_in_model([gp.predict(), lc_model(times)], map_soln)

            x_fold = (times - map_soln['Transit epoch (BTJD)'] + 0.5 * map_soln['Transit period (days)']) % map_soln[
                'Transit period (days)'
            ] - 0.5 * map_soln['Transit period (days)']
            inds = np.argsort(x_fold)
            bokeh_document.add_next_tick_callback(partial(update_initial_fit_figure, fluxes, gp_pred, inds,
                                                          lc_pred, map_soln, relative_times, times, x_fold))

            self.bokeh_document.add_next_tick_callback(partial(fit, self, map_soln, model))

        run_fitting_button.on_click(run_fitting)
        self.plot_folding_colored_lightcurve_source(initial_fit_figure, self.initial_fit_data_source,
                                                    time_column_name='Folded time (days)', flux_column_name='Relative flux')
        initial_fit_figure.line('Fit time', 'Fit', source=self.initial_fit_data_source, color='black', line_width=3)
        initial_fit_figure.sizing_mode = 'stretch_width'

        return initial_fit_figure, parameters_table

    @gen.coroutine
    def update_parameters_table(self, trace_summary):
        self.parameters_table_data_source.data = trace_summary
        self.parameters_table_data_source.data['parameter'] = trace_summary.index

    def create_folded_figured_based_on_clicks_in_unfolded_figure(self, unfolded_figure, data_source):
        # Setup empty period recording clicks for folding.
        unfolded_figure.circle('Time (BTJD)', 'Normalized PDCSAP flux', source=self.fold_coordinate_data_source,
                               color='red', alpha=0.8)  # Will be updated.
        # Prepare the folded plot.
        folded_figure = Figure(x_axis_label='Folded time (days)', y_axis_label='Normalized PDCSAP flux',
                               title=f'Folded lightcurve')
        self.plot_folding_colored_lightcurve_source(folded_figure, data_source, time_column_name='Folded time (days)')
        folded_figure.sizing_mode = 'stretch_width'
        self_ = self

        @gen.coroutine
        def update_click_dots():
            self.fold_coordinate_data_source.data = {
                'Time (BTJD)': [coordinate[0] for coordinate in self.event_coordinates],
                'Normalized PDCSAP flux': [coordinate[1] for coordinate in self.event_coordinates]
            }

        @gen.coroutine
        def update_folded_figure(folded_times):
            data_source.data['Folded time (days)'] = folded_times

        @gen.coroutine
        @without_document_lock
        def click_unfolded_figure_callback(tap_event):  # Setup what should happen when a click occurs.
            event_coordinate = tap_event.x, tap_event.y
            self.event_coordinates.append(event_coordinate)
            self.bokeh_document.add_next_tick_callback(update_click_dots)
            if len(self.event_coordinates) > 1:  # If we have more than 1 period click, we can start folding.
                event_times = [coordinate[0] for coordinate in self.event_coordinates]
                epoch, period = self.calculate_epoch_and_period_from_approximate_event_times(event_times)
                folded_times = self.fold_times(data_source.data['Time (BTJD)'], epoch, period)
                self.bokeh_document.add_next_tick_callback(partial(update_folded_figure, folded_times=folded_times))
                self_.period = period
                self_.transit_epoch = epoch
                period_depths = [coordinate[1] for coordinate in self.event_coordinates]
                self_.depth = np.abs(np.mean(period_depths))

        unfolded_figure.on_event(Tap, click_unfolded_figure_callback)
        return folded_figure

    @staticmethod
    def fold_times(times: np.ndarray, epoch: float, period: float) -> np.ndarray:
        """
        Folds an array of times based on an epoch and period.

        :param times: The times to fold.
        :param epoch: The epoch of the fold.
        :param period: The period of the fold.
        :return: The folded times.
        """
        half_period = (period / 2)
        half_period_offset_epoch_times = times - (epoch - half_period)
        half_period_offset_folded_times = half_period_offset_epoch_times % period
        folded_times = half_period_offset_folded_times - half_period
        return folded_times

    @staticmethod
    def round_series_to_significant_figures(series: pd.Series, significant_figures: int) -> pd.Series:
        """
        Rounds a series to a given number of significant figures.

        :param series: The series to round.
        :param significant_figures: The number of signficant figures to round to.
        :return: The rounded series.
        """

        def round_value_to_significant_figures(value):
            """Rounds a value to the outer scope number of significant figures"""
            return round(value, significant_figures - 1 - int(math.floor(math.log10(abs(value)))))

        return series.apply(round_value_to_significant_figures)

    @staticmethod
    def calculate_epoch_and_period_from_approximate_event_times(event_times: List[float]) -> (float, float):
        """
        Calculates the period and epoch of a signal given selected event times. The epoch is set to the first event
        chronologically.

        :param event_times: The times of the events.
        :return: The epoch and period.
        """
        sorted_event_times = np.sort(event_times)
        epoch = sorted_event_times[0]
        event_time_differences = np.diff(sorted_event_times)
        # Assume the smallest difference is close to a single period.
        smallest_time_difference = np.min(event_time_differences)
        # Get all differences close to the smallest difference to estimate a single period difference.
        threshold_from_smallest = smallest_time_difference * 0.1
        single_period_differences = event_time_differences[
            np.abs(event_time_differences - smallest_time_difference) < threshold_from_smallest]
        period_estimate_from_single_period_events = np.mean(single_period_differences)
        # Using the above estimate, estimate the number of cycles in larger time differences.
        cycles_per_time_difference = np.rint(event_time_differences / period_estimate_from_single_period_events)
        period_estimates = event_time_differences / cycles_per_time_difference
        # Weight the larger differences more heavily, based on the number of cycles estimated.
        period = np.average(period_estimates, weights=cycles_per_time_difference)
        return epoch, period

    def create_unfolded_lightcurve_figure(self, data_source):
        figure = Figure(title='Unfolded lightcurve', x_axis_label='Time (BTJD)', y_axis_label='Normalized PDCSAP flux',
                        active_drag='box_zoom')
        self.plot_folding_colored_lightcurve_source(figure, data_source)
        figure.sizing_mode = 'stretch_width'
        return figure

    def load_surrounding_lightcurves(self):
        load_range_start = self.current_target_index - self.number_of_indexes_before_and_after_to_load
        previous_indexes_to_load = [index for index in range(load_range_start, self.current_target_index)
                                    if 0 <= index < self.results_data_frame.shape[0]]
        load_range_end = self.current_target_index + self.number_of_indexes_before_and_after_to_load + 1
        subsequent_indexes_to_load = list(range(self.current_target_index + 1, load_range_end))
        indexes_to_load = [self.current_target_index] + subsequent_indexes_to_load + previous_indexes_to_load
        delete_range_start = self.current_target_index - self.number_of_indexes_before_and_after_to_delete
        delete_range_end = self.current_target_index + self.number_of_indexes_before_and_after_to_delete + 1
        indexes_to_delete = [index for index in set(range(delete_range_start, delete_range_end)) - set(indexes_to_load)
                             if 0 <= index < self.results_data_frame.shape[0]]
        for index_to_load in indexes_to_load:
            if index_to_load not in self.target_future_dictionary:
                lightcurve_path = self.results_data_frame['Lightcurve path'].iloc[index_to_load]
                target_pool_apply_result = self.pool_executor.submit(self.target_type, lightcurve_path)
                self.target_future_dictionary[index_to_load] = target_pool_apply_result
        for index_to_delete in indexes_to_delete:
            if index_to_delete in self.target_future_dictionary:
                self.target_future_dictionary.pop(index_to_delete)

    def display_current_target(self):
        target = self.target_future_dictionary[self.current_target_index].result()
        self.target_title_div.text = (f'<h1 class="title">TIC {target.tic_id} sector {target.sector}</h1>' +
                                      f'<p>Network confidence: {self.results_data_frame["Prediction"].iloc[self.current_target_index]}</p>' +
                                      f'<p>Result index: {self.current_target_index}</p>' +
                                      f'<p>Star radius (solar radii): {target.star_radius}</p>')
        if target.has_known_exofop_disposition:
            self.known_planet_div.css_classes = list(set(self.known_planet_div.css_classes) - {'is-hidden'})
        else:
            self.known_planet_div.css_classes = list(set(self.known_planet_div.css_classes).union({'is-hidden'}))
        self.lightcurve_data_source.data = {'Time (BTJD)': target.times,
                                            'Normalized PDCSAP flux': target.normalized_pdcsap_fluxes,
                                            'Normalized PDCSAP flux error': target.normalized_pdcsap_flux_errors,
                                            'Normalized SAP flux': target.normalized_sap_fluxes,
                                            'Time (days)': target.times - np.nanmin(target.times),
                                            'Folded time (days)': np.full_like(target.times, np.nan)}
        self.lightcurve_figure.x_range.start = np.min(target.times)
        self.lightcurve_figure.x_range.end = np.max(target.times)
        fluxes = target.normalized_pdcsap_fluxes
        outlier_indexes = is_outlier(fluxes, threshold=10)
        inlier_fluxes = fluxes[~outlier_indexes]
        self.lightcurve_figure.x_range.start = np.min(inlier_fluxes)
        self.lightcurve_figure.x_range.end = np.max(inlier_fluxes)
        self.fold_coordinate_data_source.data = {'Time (BTJD)': [], 'Normalized PDCSAP flux': []}
        self.event_coordinates = []
        self.initial_fit_data_source.data = {'Folded time (days)': [], 'Relative flux': [], 'Fit': [], 'Fit time': [],
                                             'Time (BTJD)': [], 'Time (days)': []}
        self.parameters_table_data_source.data = pd.DataFrame()
        self.star_radius = target.star_radius


    @staticmethod
    def create_flux_comparison_lightcurve_figure() -> (Figure, ColumnDataSource):
        figure = Figure(title='Normalized flux comparison', x_axis_label='Time (BTJD)', y_axis_label='Normalized flux',
                        active_drag='box_zoom')
        data_source = ColumnDataSource({'Time (BTJD)': [], 'Normalized PDCSAP flux': [], 'Normalized SAP flux': [],
                                        'Time (days)': [], 'Folded time (days)': []})

        def add_lightcurve(times_column_name, fluxes_column_name, legend_label, color):
            """Adds a lightcurve to the figure."""
            figure.line(source=data_source, x=times_column_name, y=fluxes_column_name, line_color=color, line_alpha=0.1)
            figure.circle(source=data_source, x=times_column_name, y=fluxes_column_name, legend_label=legend_label,
                          line_color=color, line_alpha=0.4, fill_color=color, fill_alpha=0.1)

        add_lightcurve('Time (BTJD)', 'Normalized PDCSAP flux', 'PDCSAP', 'firebrick')
        add_lightcurve('Time (BTJD)', 'Normalized SAP flux', 'SAP', 'mediumblue')
        figure.sizing_mode = 'stretch_width'
        figure.x_range = Range1d(0, 1)
        figure.y_range = Range1d(0, 1)
        return figure, data_source

    @staticmethod
    def plot_folding_colored_lightcurve_source(figure: Figure, data_source: ColumnDataSource,
                                               time_column_name: str = 'Time (BTJD)',
                                               flux_column_name: str = 'Normalized PDCSAP flux',
                                               color_value_column_name: str = 'Time (days)'):
        """
        Plots the lightcurve data source on the passed figure.

        :param figure: The figure to plot to.
        :param data_source: The data source containing the lightcurve data.
        :param time_column_name: The name of the time column whose values will be used on the x axis.
        :param flux_column_name: The name of the flux column whose values will be used on the y axis.
        :param color_value_column_name: The name of the column whose values will be used to determine data point color.
        """
        mapper = LinearColorMapper(palette='Plasma256', low=0, high=28)
        colors = {'field': color_value_column_name, 'transform': mapper}
        figure.circle(time_column_name, flux_column_name, source=data_source, fill_color=colors, fill_alpha=0.1,
                      line_color=colors, line_alpha=0.4)

    def create_target_switching_buttons(self):
        def display_next_target():
            self.current_target_index += 1
            self.load_surrounding_lightcurves()
            self.display_current_target()
        def display_previous_target():
            self.current_target_index -= 1
            self.load_surrounding_lightcurves()
            self.display_current_target()
        next_button = Button(label='Next target')
        next_button.on_click(display_next_target)
        next_button.sizing_mode = 'stretch_width'
        previous_button = Button(label='Previous target')
        previous_button.on_click(display_previous_target)
        previous_button.sizing_mode = 'stretch_width'
        return previous_button, next_button

    def create_add_to_negatives_button(self):
        add_to_negatives_button = Button(label='Add to negatives')
        negatives_csv_file_path = Path('negatives.csv')
        def add_to_negatives():
            lightcurve_path = self.results_data_frame['Lightcurve path'].iloc[self.current_target_index]
            negative_data_frame = pd.DataFrame({'Lightcurve path': [lightcurve_path]})
            if negatives_csv_file_path.exists():
                negative_data_frame.to_csv(negatives_csv_file_path, mode='a', header=False)
            else:
                negative_data_frame.to_csv(negatives_csv_file_path)
        add_to_negatives_button.on_click(add_to_negatives)
        return add_to_negatives_button
