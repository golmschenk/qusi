"""Script that generates a Gravitational Microlensing Signal, randomly, within the natural parameters: u0 (
source-lens impact parameter), tE (Einstein radius crossing time), rho (angular source size normalized by the
angular Einstein radius) , s (Projected separation of the masses normalized by the angular Einstein radius),
q (Mass ratio M_planet/M_host), alpha (Trajectory angle).
"""
try:
    import muLAn.models.BLcontU as esbl_vbb
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(f'{__file__} module requires the muLAn package. Please install separately.') from error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MagnificationSignal:
    def __init__(self, directory_path_with_te_and_rho='distribution/'):
        """

        :param directory_path_with_te_and_rho:
        """
        self.path = directory_path_with_te_and_rho
        self.n_data_points = 40000
        self.timeserie = np.linspace(-30, 30, self.n_data_points)
        self.magnification = None
        self.magnification_signal_curve = None
        self.u0 = None
        self.tE = None
        self.rho = None
        self.s = None
        self.q = None
        self.alpha = None

    def getting_random_values(self):
        """
        Set randomly the natural parameters: u0 (source-lens impact parameter), tE (Einstein radius crossing time),
        rho (angular source size normalized by the angular Einstein radius) , s (Projected separation of the masses
        normalized by the angular Einstein radius), q (Mass ratio M_planet/M_host), alpha (Trajectory angle)
        :return:
        u0
        tE
        rho
        s
        q
        alpha
        """
        u0_list = np.linspace(0.01, 3.5, 10)
        self.u0 = np.random.choice(u0_list)

        tE_path = self.path + 'tE_distribution.feather'
        tE_list = pd.read_feather(tE_path)
        tE_list = (tE_list['fake_tE']).to_numpy()
        self.tE = np.random.choice(tE_list)

        rho_path = self.path + 'rho_distribution.feather'
        rho_list = pd.read_feather(rho_path)
        rho_list = (rho_list['fake_rho']).to_numpy()
        self.rho = np.random.choice(rho_list)

        s_list = np.linspace(0.01, 3.5, 100)
        self.s = np.random.choice(s_list)

        q_list = np.linspace(0.0001, 10000, 100000)
        self.q = np.random.choice(q_list)

        alpha_list = np.linspace(0, np.pi, 10)
        self.alpha = np.random.choice(alpha_list)

        return self.u0, self.tE, self.rho, self.s, self.q, self.alpha

    def generating_magnification(self):
        """
        Creates the magnification signal
        :return:
        magnification_signal_curve: Pandas data frame with time and magnification
        """
        lens_params = dict({'u0': self.u0,
                            'tE': self.tE,
                            't0': 0.0,
                            'piEN': 0.0,
                            'piEE': 0.0,
                            'rho': self.rho,
                            's': self.s,
                            'q': self.q,
                            'alpha': self.alpha,
                            'dadt': 0.0,
                            'dsdt': 0.0,
                            })
        # No lens orbital motion (dalpha=0, ds=0)

        tb = lens_params['t0']  # we choose t_binary = t0 here (see, e.g., Skowron et al. 2011)

        # We don't want to include microlens parallax in this fit
        Ds = dict({'N': np.zeros(self.n_data_points), 'E': np.zeros(self.n_data_points)})

        # Compute magnification
        self.magnification = esbl_vbb.magnifcalc(self.timeserie, lens_params, Ds=Ds, tb=tb)
        self.magnification_signal_curve = pd.DataFrame({'Time': self.timeserie, 'Magnification': self.magnification})
        return self.magnification_signal_curve

    def plot_magnification(self):
        """
        Plot the lightcurve.
        :return:
        """
        # PLOT
        plt.plot(self.timeserie, self.magnification)
        plt.xlabel('Days')
        plt.ylabel('Magnification')
        plt.title(f'u0= {self.u0:3.5f}; tE= {self.tE:12.5f}; rho= {self.rho:8.5f};\n s= {self.s:3.5f}; '
                  f'q= {self.q:8.5f}; alpha= {self.alpha:3.5f}')
        # plt.title(f'u0= {:3.5f}; tE= {:10.5f}; rho= {:5.5f};\n s= {:3.5f}; '
        #           f'q= {:5.5f}; alpha= {:2.5f}'.format(self.u0,self.tE,self.rho,self.s,self.q,self.alpha))
        plt.show()
        return None


if __name__ == '__main__':
    microlensing_signal = MagnificationSignal()
    microlensing_signal.getting_random_values()
    magnification_curve = microlensing_signal.generating_magnification()
    microlensing_signal.plot_magnification()
    print("Done")
