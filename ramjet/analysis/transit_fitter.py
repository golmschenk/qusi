"""Code for producing a transit fitting."""
import itertools
import math
from typing import List

import numpy as np
import pandas as pd
from bokeh.events import Tap
from bokeh.models import Column, ColumnDataSource, LinearColorMapper, Button, DataTable, TableColumn
from bokeh.plotting import Figure
from bokeh.server.server import Server
import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
# import gspread
# from google.oauth2.service_account import Credentials

from ramjet.data_interface.tess_data_interface import TessDataInterface


class TransitFitter:
    """
    A class to fit a transit.
    """

    def __init__(self, tic_id, sectors=None):
        tess_data_interface = TessDataInterface()
        self.title = f'TIC {tic_id}'
        relative_fluxes_arrays = []
        relative_flux_errors_arrays = []
        times_arrays = []
        dispositions_data_frame = tess_data_interface.retrieve_exofop_toi_and_ctoi_planet_disposition_for_tic_id(tic_id)
        if dispositions_data_frame.shape[0] == 0:
            print('No known ExoFOP dispositions found.')
        # Use context options to not truncate printed data.
        else:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                print(dispositions_data_frame)
                exit()
        if sectors is None:
            sectors = tess_data_interface.get_sectors_target_appears_in(tic_id)
        if isinstance(sectors, int):
            sectors = [sectors]
        for sector in sectors:
            lightcurve_path = tess_data_interface.download_lightcurve(tic_id, sector)
            lightcurve = tess_data_interface.load_fluxes_flux_errors_and_times_from_fits_file(lightcurve_path)
            sector_fluxes, sector_flux_errors, sector_times = lightcurve
            sector_flux_median = np.median(sector_fluxes)
            sector_normalized_fluxes = sector_fluxes / sector_flux_median - 1
            sector_normalized_flux_errors = sector_flux_errors / sector_flux_median
            relative_fluxes_arrays.append(sector_normalized_fluxes)
            relative_flux_errors_arrays.append(sector_normalized_flux_errors)
            times_arrays.append(sector_times)
        self.sectors = sectors
        self.tic_id = tic_id
        self.times = np.concatenate(times_arrays).astype(np.float64)
        self.relative_fluxes = np.concatenate(relative_fluxes_arrays).astype(np.float64)
        self.relative_flux_errors = np.concatenate(relative_flux_errors_arrays).astype(np.float64)
        tic_row = tess_data_interface.get_tess_input_catalog_row(tic_id)
        self.star_radius = tic_row['rad']
        print(self.star_radius)
        self.period = None
        self.depth = None
        self.transit_epoch = None

    def bokeh_application(self, bokeh_document):
        lightcurve_figure = self.create_lightcurve_figure()
        folded_figure = self.add_folded_figured_based_on_clicks_in_unfolded_figure(lightcurve_figure)
        run_fitting_button = Button(label='Run fitting')
        initial_fit_figure, parameters_table = self.create_mcmc_fit_figures(run_fitting_button)
        column = Column(lightcurve_figure, folded_figure, run_fitting_button, initial_fit_figure, parameters_table)
        column.sizing_mode = 'stretch_width'
        bokeh_document.add_root(column)

    def create_lightcurve_figure(self):
        figure = Figure(title=self.title, x_axis_label='Time (days)', y_axis_label='Relative flux')

        def draw_lightcurve(times, fluxes):
            data_source = ColumnDataSource({'Time (days)': times, 'Relative flux': fluxes})
            mapper = LinearColorMapper(
                palette='Plasma256',
                low=np.min(data_source.data['Time (days)']),
                high=np.max(data_source.data['Time (days)'])
            )
            colors = {'field': 'Time (days)', 'transform': mapper}
            figure.circle('Time (days)', 'Relative flux',
                          fill_color=colors, fill_alpha=0.1, line_color=colors, line_alpha=0.4,
                          source=data_source)

        draw_lightcurve(self.times, self.relative_fluxes)
        figure.sizing_mode = 'stretch_width'
        return figure

    def add_folded_figured_based_on_clicks_in_unfolded_figure(self, unfolded_figure):
        # Setup empty period recording clicks for folding.
        event_coordinates = []
        event_coordinates_data_source = ColumnDataSource({'Time (days)': [], 'Relative flux': []})
        unfolded_figure.circle('Time (days)', 'Relative flux', source=event_coordinates_data_source,
                               color='red', alpha=0.8)  # Will be updated.
        # Prepare the folded plot.
        folded_data_source = ColumnDataSource({'Relative flux': self.relative_fluxes,
                                               'Folded time (days)': [],
                                               'Time (days)': self.times})
        folded_figure = Figure(x_axis_label='Folded time (days)', y_axis_label='Relative flux',
                               title=f'Folded {self.title}')
        mapper = LinearColorMapper(
            palette='Plasma256',
            low=np.min(self.times),
            high=np.max(self.times)
        )
        colors = {'field': 'Time (days)', 'transform': mapper}
        folded_figure.circle('Folded time (days)', 'Relative flux',
                             fill_color=colors, fill_alpha=0.1, line_color=colors, line_alpha=0.4,
                             source=folded_data_source)
        folded_figure.sizing_mode = 'stretch_width'
        self_ = self

        def click_unfolded_figure_callback(tap_event):  # Setup what should happen when a click occurs.
            event_coordinate = tap_event.x, tap_event.y
            event_coordinates.append(event_coordinate)
            event_coordinates_data_source.data = {
                'Time (days)': [coordinate[0] for coordinate in event_coordinates],
                'Relative flux': [coordinate[1] for coordinate in event_coordinates]
            }
            if len(event_coordinates) > 1:  # If we have more than 1 period click, we can start folding.
                event_times = [coordinate[0] for coordinate in event_coordinates]
                epoch, period = self.calculate_epoch_and_period_from_approximate_event_times(event_times)
                folded_times = self.fold_times(self_.times, epoch, period)
                folded_data_source.data['Folded time (days)'] = folded_times
                # folded_figure.x_range.start = -period/10
                # folded_figure.x_range.end = period/10
                self_.period = period
                self_.transit_epoch = epoch
                period_depths = [coordinate[1] for coordinate in event_coordinates]
                self_.depth = np.abs(np.mean(period_depths))

        unfolded_figure.on_event(Tap, click_unfolded_figure_callback)
        return folded_figure

    def create_mcmc_fit_figures(self, run_fitting_button):
        initial_fit_data_source = ColumnDataSource({'Folded time (days)': [], 'Relative flux': [],
                                                    'Fit': [], 'Fit time': [], 'Time (BTJD)': self.times})
        self_ = self
        initial_fit_figure = Figure(x_axis_label='Folded time (days)', y_axis_label='Relative flux',
                                    title=f'Initial fit {self.title}')
        parameters_table_data_source = ColumnDataSource(pd.DataFrame())
        parameters_table_columns = [TableColumn(field=column, title=column) for column in
                                    ['parameter', 'mean', 'sd', 'r_hat']]
        parameters_table = DataTable(source=parameters_table_data_source, columns=parameters_table_columns,
                                     editable=True)

        def run_fitting():
            with pm.Model() as model:
                # Stellar parameters
                mean = pm.Normal("mean", mu=0.0, sigma=10.0 * 1e-3)
                u = xo.distributions.QuadLimbDark("u")
                star_params = [mean, u]

                # Gaussian process noise model
                sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.median(self_.relative_flux_errors))
                log_Sw4 = pm.Normal("log_Sw4", mu=0.0, sigma=10.0)
                log_w0 = pm.Normal("log_w0", mu=np.log(2 * np.pi / 10.0), sigma=10.0)
                kernel = xo.gp.terms.SHOTerm(log_Sw4=log_Sw4, log_w0=log_w0, Q=1.0 / 3)
                noise_params = [sigma, log_Sw4, log_w0]

                # Planet parameters
                log_ror = pm.Normal("log_ror", mu=0.5 * np.log(self_.depth), sigma=10.0 * 1e-3)
                ror = pm.Deterministic("ror", tt.exp(log_ror))
                depth = pm.Deterministic("depth", tt.square(ror))

                # Orbital parameters
                log_period = pm.Normal("log_period", mu=np.log(self_.period), sigma=1.0)
                t0 = pm.Normal("t0", mu=self_.transit_epoch, sigma=1.0)
                log_dur = pm.Normal("log_dur", mu=np.log(0.1), sigma=10.0)
                b = xo.distributions.ImpactParameter("b", ror=ror)

                period = pm.Deterministic("period", tt.exp(log_period))
                dur = pm.Deterministic("dur", tt.exp(log_dur))

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
                gp = xo.gp.GP(kernel, self_.times, (self_.relative_flux_errors ** 2) + (sigma ** 2), mean=lc_model)
                gp.marginal("obs", observed=self_.relative_fluxes)

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
                gp_pred, lc_pred = xo.eval_in_model([gp.predict(), lc_model(self_.times)], map_soln)

            x_fold = (self_.times - map_soln["t0"] + 0.5 * map_soln["period"]) % map_soln[
                "period"
            ] - 0.5 * map_soln["period"]
            inds = np.argsort(x_fold)
            initial_fit_data_source.data['Folded time (days)'] = x_fold
            initial_fit_data_source.data['Relative flux'] = self_.relative_fluxes - gp_pred - map_soln["mean"]
            initial_fit_data_source.data['Fit'] = lc_pred[inds] - map_soln["mean"]
            initial_fit_data_source.data['Fit time'] = x_fold[
                inds]  # TODO: This is terrible, you should be able to line them up *afterward* to not make a duplicate time column

            with model:
                trace = pm.sample(
                    tune=2000,
                    draws=2000,
                    start=map_soln,
                    chains=4,
                    step=xo.get_dense_nuts_step(target_accept=0.9),
                )

            trace_summary = pm.summary(trace, round_to='none')  # Not a typo. PyMC3 wants 'none' as a string here.
            epoch = round(trace_summary['mean']['t0'], 3)  # Round the epoch differently, as BTJD needs more digits.
            trace_summary['mean'] = self_.round_series_to_significant_figures(trace_summary['mean'], 5)
            trace_summary['mean']['t0'] = epoch
            parameters_table_data_source.data = trace_summary
            parameters_table_data_source.data['parameter'] = trace_summary.index
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
            # worksheet.update_cell(empty_row_index, 3, trace_summary['mean']['t0'])
            # worksheet.update_cell(empty_row_index, 4, trace_summary['mean']['period'])
            # worksheet.update_cell(empty_row_index, 5, trace_summary['mean']['depth'])
            # worksheet.update_cell(empty_row_index, 6, trace_summary['mean']['dur'])
            # worksheet.update_cell(empty_row_index, 7, self_.star_radius)
            # worksheet.update_cell(empty_row_index, 8, trace_summary['mean']['ror'] * self_.star_radius)

        run_fitting_button.on_click(run_fitting)
        self.plot_lightcurve_source(initial_fit_figure, initial_fit_data_source, time_column_name='Folded time (days)')
        initial_fit_figure.line('Fit time', 'Fit', source=initial_fit_data_source, color='black', line_width=3)
        initial_fit_figure.sizing_mode = 'stretch_width'

        return initial_fit_figure, parameters_table

    def plot_lightcurve_source(self, figure: Figure, data_source: ColumnDataSource,
                               time_column_name: str = 'Time (BTJD)',
                               flux_column_name: str = 'Relative flux',
                               color_value_column_name: str = 'Time (BTJD)'):
        mapper = LinearColorMapper(palette='Plasma256',
                                   low=np.min(data_source.data[color_value_column_name]),
                                   high=np.max(data_source.data[color_value_column_name]))
        colors = {'field': color_value_column_name, 'transform': mapper}
        figure.circle(time_column_name, flux_column_name, source=data_source, fill_color=colors, fill_alpha=0.1,
                      line_color=colors, line_alpha=0.4)

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


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    # Start the server.
    server = Server({'/': TransitFitter(tic_id=362043085).bokeh_application})
    server.start()
    # Start the specific application on the server.
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
