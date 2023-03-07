"""
NOT READY TO USE
Script that generates a Gravitational Microlensing Signal, randomly, within the natural parameters: u0 (
source-lens impact parameter), tE (Einstein radius crossing time), rho (angular source size normalized by the
angular Einstein radius) , s (Projected separation of the masses normalized by the angular Einstein radius),
q (Mass ratio M_planet/M_host), alpha (Trajectory angle). The distribution for tE and rho are based on the MOA
observations.
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from muLAn.models.vbb.vbb import vbbmagU
except ModuleNotFoundError as error:
    vbbmagU = None

# Things I know from Iona's draft paper:
# source star 10.5 < H < 22
# lens star - infinity < H < infinity

# 0 ≤ t0 ≤ Tobs , Tobs for PRIME: 365.25 days
# 0 ≤ u0 ≤ u0_max, u0_max = 1.0

# lens-source relative parallax : 1 (1/D_L − 1/D_S)
def lens_source_relative_parallax_calculator(distance_lens__au, distance_star__au):
    lens_source_relative_parallax = (1 / distance_lens__au - 1 / distance_star__au)
    return lens_source_relative_parallax

# angular einstein radius
def angular_Einstein_ring_radius_calculator(mass_lens__msun, distance_lens__au, distance_star__au):
    G_constant__pc_km2_s2_sun_mass = 4.3009 * 10**(-3) # pc (km/s)^2 / M_sun
    speed_of_light_km_s = 300000
    # 300,000 km/s
    lens_source_relative_parallax = lens_source_relative_parallax_calculator(distance_lens__au, distance_star__au)
    angular_Einstein_ring_radius = np.sqrt(4 * G_constant__pc_km2_s2_sun_mass *
                                           mass_lens__msun * lens_source_relative_parallax / (speed_of_light_km_s**2))
    return angular_Einstein_ring_radius

# The angular radius of the source star θ∗ = R∗/DS, where R∗ is the radius of the source star estimated
# from the source magnitude from genstars.
def angular_radius_of_the_source_star_calculator(estimated_radius_source_star, distance_star__au):
    angular_radius_of_the_source_star = estimated_radius_source_star / distance_star__au
    return angular_radius_of_the_source_star

# The angular source size (Source angular radius) θstar normalized by the angular Einstein radius θE
def angular_Einstein_radius_rho_calculator(angular_Einstein_ring_radius, source_angular_radius):
    angular_Einstein_radius_rho = source_angular_radius / angular_Einstein_ring_radius
    return angular_Einstein_radius_rho

# I don't have tE neither Tstar
# tE
def einstein_crossing_time_calculator(angular_Einstein_radius_rho):
    tE = t_star / rho

    t_star = rho * tE

# lens-source relative proper motion θE/tE or θstar/tstar
def lens_source_relative_proper_motion_calculator(angular_Einstein_ring_radius):
    lens_source_relative_proper_motion = angular_Einstein_ring_radius /





# to add a planet
# (0.1 < Mp < 10^4) M_earth
# (3 < a < 30) au


class MagnificationSignal:
    """A class to generate a random microlensing magnification signal.
    Using the parameters:
    u0 (source-lens impact parameter)
    tE (Einstein radius crossing time)
    rho (angular source size normalized by the angular Einstein radius)
    s (Projected separation of the masses normalized by the angular Einstein radius)
    q (Mass ratio M_planet/M_host)
    alpha (Trajectory angle)
    > The distribution for tE and rho are based on the MOA observations
    > No parallax effect is considered
    """
    def __init__(self):
        self.load_moa_meta_data_to_class_attributes()
        self.n_data_points = 80000
        self.timeseries = np.linspace(-30, 30, self.n_data_points)
        self.magnification = None
        self.magnification_signal_curve = None
        self.u0 = None
        self.tE = None
        self.rho = None
        self.s = None
        self.q = None
        self.alpha = None


        # Compute magnification
        self.magnification = self.calculating_magnification_from_vbb(self.timeseries, lens_params)
        self.magnification_signal_curve = pd.DataFrame({'Time': self.timeseries, 'Magnification': self.magnification})

    def plot_magnification(self):
        """
        Plot the light curve.
        """
        # PLOT
        plt.plot(self.timeseries, self.magnification)
        plt.xlabel('Days')
        plt.ylabel('Magnification')
        plt.title(f'u0= {self.u0:3.5f}; tE= {self.tE:12.5f}; rho= {self.rho:8.5f};\n s= {self.s:3.5f}; '
                  f'q= {self.q:8.5f}; alpha= {self.alpha:3.5f}')
        # plt.title(f'u0= {:3.5f}; tE= {:10.5f}; rho= {:5.5f};\n s= {:3.5f}; '
        #           f'q= {:5.5f}; alpha= {:2.5f}'.format(self.u0,self.tE,self.rho,self.s,self.q,self.alpha))
        plt.show()

    @classmethod
    def generate_randomly_based_on_moa_observations(cls, time_range: float = 30):
        microlensing_signal = cls()
        microlensing_signal.timeseries = np.linspace(-time_range, time_range, microlensing_signal.n_data_points)
        microlensing_signal.getting_random_values()
        microlensing_signal.generating_magnification()
        return microlensing_signal

    @classmethod
    def generate_approximately_pspl_randomly_based_on_moa_observations(cls, time_range: float = 30):
        microlensing_signal = cls()
        microlensing_signal.timeseries = np.linspace(-time_range, time_range, microlensing_signal.n_data_points)
        microlensing_signal.getting_random_values()
        microlensing_signal.q = 0.00001
        microlensing_signal.s = 10
        microlensing_signal.alpha = np.pi / 2
        microlensing_signal.generating_magnification()
        return microlensing_signal

    @staticmethod
    def calculating_magnification_from_vbb(timeseries, lens_params):
        """Return the VBB method finite-source uniform magnification.
        Adapted from muLAn: gravitational MICROlensing Analysis Software.
        """
        # Get parameters
        t0 = lens_params['t0']
        u0 = lens_params['u0']
        tE = lens_params['tE']
        rho = lens_params['rho']
        q = lens_params['q']
        alpha = lens_params['alpha']
        s = lens_params['s']

        tau = (timeseries - t0) / tE

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        x = tau * cos_alpha - u0 * sin_alpha
        y = tau * sin_alpha + u0 * cos_alpha

        # Conversion secondary body left -> right
        x = -x
        # Compute magnification
        accuracy = 1.e-3  # Absolute mag accuracy (mag+/-accuracy)
        magnification = np.array([vbbmagU(s, q, rho, x[i], y[i], accuracy) for i in range(len(x))])
        return magnification


if __name__ == '__main__':
    import time
    start_time = time.time()
    random_signal = MagnificationSignal.generate_randomly_based_on_moa_observations()
    print("--- %s seconds ---" % (time.time() - start_time))
    random_signal.plot_magnification()
    print("Done")
