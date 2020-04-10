"""Code for producing a transit fitting."""
import numpy as np
import pandas as pd
from bokeh.events import Tap
from bokeh.models import Column, ColumnDataSource, LinearColorMapper, Button, DataTable, TableColumn
from bokeh.plotting import Figure
from bokeh.server.server import Server
import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo

from ramjet.data_interface.tess_data_interface import TessDataInterface, TessFluxType


class TransitFitter:
    def __init__(self, tic_id, sector):
        self.tess_data_interface = TessDataInterface()
        lightcurve_path = self.tess_data_interface.download_lightcurve(tic_id, sector)
        self.title = f'TIC {tic_id} sector {sector}'
        pdcsap_fluxes, pdcsap_flux_errors, pdcsap_times = self.tess_data_interface.load_fluxes_flux_errors_and_times_from_fits_file(
            lightcurve_path)
        pdcsap_median = np.median(pdcsap_fluxes)
        self.times = pdcsap_times.astype(np.float64)
        self.normalized_fluxes = (pdcsap_fluxes.astype(np.float64) / pdcsap_median) - 1
        self.normalized_flux_errors = pdcsap_flux_errors.astype(np.float64) / pdcsap_median
        tic_row = self.tess_data_interface.get_tess_input_catalog_row(tic_id)
        self.star_radius = tic_row['rad']
        self.period = None
        self.depth = None
        self.transit_epoch = None

    def bokeh_application(self, bokeh_document):
        lightcurve_figure = self.create_lightcure_figure()
        folded_figure = self.add_folded_figured_based_on_clicks_in_unfolded_figure(lightcurve_figure)
        run_fitting_button = Button(label='Run fitting')
        initial_fit_figure, parameters_table = self.create_mcmc_fit_figures(run_fitting_button)
        column = Column(lightcurve_figure, folded_figure, run_fitting_button, initial_fit_figure, parameters_table)
        column.sizing_mode = 'stretch_width'
        bokeh_document.add_root(column)

    def create_lightcure_figure(self):
        figure = Figure(title=self.title, x_axis_label='Time (days)', y_axis_label='Relative flux')

        def draw_lightcurve(times, fluxes, legend_label):
            data_source = ColumnDataSource({'Time (days)': times, 'Relative flux': fluxes})
            mapper = LinearColorMapper(
                palette='Plasma256',
                low=np.min(times),
                high=np.max(times)
            )
            colors = {'field': 'Time (days)', 'transform': mapper}
            figure.circle('Time (days)', 'Relative flux',
                          fill_color=colors, fill_alpha=0.1, line_color=colors, line_alpha=0.4,
                          source=data_source)

        draw_lightcurve(self.times, self.normalized_fluxes, 'PDCSAP')
        figure.sizing_mode = 'stretch_width'
        return figure

    def add_folded_figured_based_on_clicks_in_unfolded_figure(self, unfolded_figure):
        # Setup empty period recording clicks for folding.
        period_coordinates = []
        period_coordinates_data_source = ColumnDataSource({'Time (days)': [], 'Relative flux': []})
        unfolded_figure.circle('Time (days)', 'Relative flux', source=period_coordinates_data_source,
                               color='red', alpha=0.8)  # Will be updated.
        # Prepare the folded plot.
        folded_data_source = ColumnDataSource({'Relative flux': self.normalized_fluxes,
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
            period_coordinate = tap_event.x, tap_event.y
            period_coordinates.append(period_coordinate)
            period_coordinates_data_source.data = {
                'Time (days)': [coordinate[0] for coordinate in period_coordinates],
                'Relative flux': [coordinate[1] for coordinate in period_coordinates]
            }
            if len(period_coordinates) > 1:  # If we have more than 1 period click, we can start folding.
                period_times = [coordinate[0] for coordinate in period_coordinates]
                period_times = sorted(period_times)
                period_times = np.array(period_times, dtype=np.float64)
                period = np.mean(np.diff(period_times))  # Assumes periods are contiguous (don't jump over one).
                epoch_times = self_.times - (period_times[0] - (period / 2))
                folded_pdcsap_times = epoch_times % period - (period / 2)
                folded_data_source.data['Folded time (days)'] = folded_pdcsap_times
                # folded_figure.x_range.start = -period/10
                # folded_figure.x_range.end = period/10
                self_.period = period
                self_.transit_epoch = period_times[0]
                period_depths = [coordinate[1] for coordinate in period_coordinates]
                self_.depth = np.abs(np.mean(period_depths))

        unfolded_figure.on_event(Tap, click_unfolded_figure_callback)
        return folded_figure

    def create_mcmc_fit_figures(self, run_fitting_button):
        initial_fit_data_source = ColumnDataSource({'Folded time (days)': [], 'Relative flux': [],
                                                    'Fit': [], 'Fit time': [], 'Time (days)': self.times})
        self_ = self
        initial_fit_figure = Figure(x_axis_label='Folded time (days)', y_axis_label='Relative flux',
                                    title=f'Initial fit {self.title}')
        parameters_table_data_source = ColumnDataSource(pd.DataFrame())
        parameters_table_columns = [TableColumn(field=column, title=column) for column in ['parameter', 'mean', 'sd', 'r_hat']]
        parameters_table = DataTable(source=parameters_table_data_source, columns=parameters_table_columns)
        def run_fitting():
            with pm.Model() as model:
                # Stellar parameters
                mean = pm.Normal("mean", mu=0.0, sigma=10.0 * 1e-3)
                u = xo.distributions.QuadLimbDark("u")
                star_params = [mean, u]

                # Gaussian process noise model
                sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.median(self_.normalized_flux_errors))
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
                        star.get_light_curve(orbit=orbit, r=ror*self.star_radius, t=t), axis=-1
                    )

                # Finally the GP observation model
                gp = xo.gp.GP(kernel, self_.times, self_.normalized_flux_errors ** 2 + sigma ** 2, mean=lc_model)
                gp.marginal("obs", observed=self_.normalized_fluxes)

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
            initial_fit_data_source.data['Relative flux'] = self_.normalized_fluxes - gp_pred - map_soln["mean"]
            initial_fit_data_source.data['Fit'] = lc_pred[inds] - map_soln["mean"]
            initial_fit_data_source.data['Fit time'] = x_fold[inds]  # TODO: This is terrible, you should be able to line them up *afterward* to not make a duplicate time column

            with model:
                trace = pm.sample(
                    tune=2000,
                    draws=2000,
                    start=map_soln,
                    chains=4,
                    step=xo.get_dense_nuts_step(target_accept=0.9),
                )

            trace_summary = pm.summary(trace)
            parameters_table_data_source.data = trace_summary
            parameters_table_data_source.data['parameter'] = trace_summary.index
            with pd.option_context('precision', 10, 'display.max_columns', None, 'display.max_rows', None):
                print(trace_summary)

        run_fitting_button.on_click(run_fitting)
        mapper = LinearColorMapper(
            palette='Plasma256',
            low=np.min(self.times),
            high=np.max(self.times)
        )
        colors = {'field': 'Time (days)', 'transform': mapper}
        initial_fit_figure.circle('Folded time (days)', 'Relative flux', source=initial_fit_data_source,
                                  fill_color=colors, fill_alpha=0.1, line_color=colors, line_alpha=0.4)
        initial_fit_figure.line('Fit time', 'Fit', source=initial_fit_data_source, color='black', line_width=3)
        initial_fit_figure.sizing_mode = 'stretch_width'

        return initial_fit_figure, parameters_table


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    # Start the server.
    server = Server({'/': TransitFitter(tic_id=362043085, sector=5).bokeh_application})
    server.start()
    # Start the specific application on the server.
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
