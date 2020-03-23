"""Script that generates a Gravitational Microlensing Signal, randomly, within the natural parameters: u0 (
source-lens impact parameter), tE (Einstein radius crossing time), rho (angular source size normalized by the
angular Einstein radius) , s (Projected separation of the masses normalized by the angular Einstein radius),
q (Mass ratio M_planet/M_host), alpha (Trajectory angle). The distribution for tE and rho are based on the MOA
observations.
"""
try:
    from muLAn.models.vbb.vbb import vbbmagU
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(f'{__file__} module requires the muLAn package. Please install separately.') from error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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

# --------------------------------------------------------------------


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
    tE_list: pd.Series = None
    rho_list: pd.Series = None

    def __init__(self):
        self.load_moa_meta_data_to_class_attributes()
        self.n_data_points = 40000
        self.timeseries = np.linspace(-30, 30, self.n_data_points)
        self.magnification = None
        self.magnification_signal_curve = None
        self.u0 = None
        self.tE = None
        self.rho = None
        self.s = None
        self.q = None
        self.alpha = None

    def load_moa_meta_data_to_class_attributes(self):
        """
        Loads the MOA meta data defining microlensing to class attributes. If already loaded, does nothing.
        """
        if self.tE_list is None:
            microlensing_meta_data_path = Path(__file__).parent.joinpath(
                'microlensing_signal_meta_data/moa9yr_events_meta_oct2018.txt')
            try:
                df = pd.read_csv(microlensing_meta_data_path, header=None, skipinitialspace=True, names=['event'])
            except FileNotFoundError as error_:
                raise FileNotFoundError(f'{microlensing_meta_data_path} is required\nMOA metadata not found.' +
                                        'Please, contact the Microlensing Group to get this file') from error_

            data = df['event'].str.split("\s+", 134, expand=True)
            self.tE_list = data[58]
            self.rho_list = data[107]

    def getting_random_values(self):
        """
        Set randomly the natural parameters: u0 (source-lens impact parameter), tE (Einstein radius crossing time),
        rho (angular source size normalized by the angular Einstein radius) , s (Projected separation of the masses
        normalized by the angular Einstein radius), q (Mass ratio M_planet/M_host), alpha (Trajectory angle)
        """

        u0_list = np.linspace(-3.5, 3.5, 1000)
        self.u0 = np.random.choice(u0_list)

        self.tE = float(np.random.choice(self.tE_list))

        self.rho = float(np.random.choice(self.rho_list))

        s_list = np.linspace(0.01, 3.5, 100)
        self.s = np.random.choice(s_list)

        q_list = np.power(10, (np.linspace(-5, 0, 3000)))
        self.q = np.random.choice(q_list)

        alpha_list = np.linspace(0, 2 * np.pi, 60)
        self.alpha = np.random.choice(alpha_list)

    def generating_magnification(self):
        """
        Creates the magnification signal
        """
        lens_params = dict({'u0': self.u0,
                            'tE': self.tE,
                            't0': 0.0,
                            'rho': self.rho,
                            's': self.s,
                            'q': self.q,
                            'alpha': self.alpha
                            })

        # Compute magnification
        self.magnification = calculating_magnification_from_vbb(self.timeseries, lens_params)
        self.magnification_signal_curve = pd.DataFrame({'Time': self.timeseries, 'Magnification': self.magnification})

    def plot_magnification(self):
        """
        Plot the lightcurve.
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
    def generate_randomly_based_on_moa_observations(cls):
        microlensing_signal = cls()
        microlensing_signal.getting_random_values()
        microlensing_signal.generating_magnification()
        return microlensing_signal


if __name__ == '__main__':
    import time
    start_time = time.time()
    random_signal = MagnificationSignal.generate_randomly_based_on_moa_observations()
    print("--- %s seconds ---" % (time.time() - start_time))
    random_signal.plot_magnification()
    print("Done")
