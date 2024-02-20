"""Script that generates a Gravitational Microlensing Signal, randomly, within the natural parameters: u0 (
source-lens impact parameter), tE (Einstein radius crossing time), rho (angular source size normalized by the
angular Einstein radius) , s (Projected separation of the masses normalized by the angular Einstein radius),
q (Mass ratio M_planet/M_host), alpha (Trajectory angle). The distribution for tE and rho are based on the MOA
observations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    from muLAn.models.vbb.vbb import vbbmagU
except ModuleNotFoundError:

    def vbbmagU(_s, _q, _rho, _xi, _yi, _accuracy):  # noqa
        raise ModuleNotFoundError


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

    einstein_crossing_time_list: np.ndarray = None
    rho_list: np.ndarray = None

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

    def load_moa_meta_data_to_class_attributes(self):
        """
        Loads the MOA meta data defining microlensing to class attributes. If already loaded, does nothing.
        """
        if self.einstein_crossing_time_list is None:
            microlensing_meta_data_path = Path(__file__).parent.joinpath(
                "microlensing_signal_meta_data/candlist_RADec.dat.txt"
            )
            microlensing_meta_data_path.parent.mkdir(parents=True, exist_ok=True)
            if not microlensing_meta_data_path.exists():
                candidate_list_csv_url = "https://exoplanetarchive.ipac.caltech.edu/data/ExoData/MOA/candlist_RADec.dat"
                response = requests.get(candidate_list_csv_url, timeout=600)
                with open(microlensing_meta_data_path, "wb") as csv_file:
                    csv_file.write(response.content)
            data = pd.read_csv(
                microlensing_meta_data_path,
                header=None,
                delim_whitespace=True,
                comment="#",
                usecols=[19, 36],
                names=["tE", "rho"],
            )
            self.einstein_crossing_time_list: np.ndarray = data["tE"].values
            self.rho_list: np.ndarray = data["rho"].values
            bad_einstein_crossing_time = 6000
            bad_indexes = np.argwhere(self.einstein_crossing_time_list > bad_einstein_crossing_time)
            self.einstein_crossing_time_list = np.delete(self.einstein_crossing_time_list, bad_indexes)
            self.rho_list = np.delete(self.rho_list, bad_indexes)

    def getting_random_values(self):
        """
        Set randomly the natural parameters: u0 (source-lens impact parameter), tE (Einstein radius crossing time),
        rho (angular source size normalized by the angular Einstein radius) , s (Projected separation of the masses
        normalized by the angular Einstein radius), q (Mass ratio M_planet/M_host), alpha (Trajectory angle)
        """

        u0_list = np.linspace(-0.1, 0, 1000)
        self.u0 = np.random.choice(u0_list)

        index = np.random.choice(np.arange(self.einstein_crossing_time_list.shape[0]))
        self.tE = float(self.einstein_crossing_time_list[index])
        self.rho = float(self.rho_list[index])

        s_list = np.linspace(0.7, 1.3, 100)
        self.s = np.random.choice(s_list)

        q_list = np.power(10, (np.linspace(-2.5, -0.3, 1000)))
        self.q = np.random.choice(q_list)

        pi_denomintor = 128
        alpha_list = np.concatenate(
            [np.linspace(0, np.pi / pi_denomintor), np.linspace(np.pi - (np.pi / pi_denomintor), np.pi)]
        )
        self.alpha = np.random.choice(alpha_list)

    def generating_magnification(self):
        """
        Creates the magnification signal
        """
        lens_params = {
            "u0": self.u0,
            "tE": self.tE,
            "t0": 0.0,
            "rho": self.rho,
            "s": self.s,
            "q": self.q,
            "alpha": self.alpha,
        }

        # Compute magnification
        self.magnification = self.calculating_magnification_from_vbb(self.timeseries, lens_params)
        self.magnification_signal_curve = pd.DataFrame({"Time": self.timeseries, "Magnification": self.magnification})

    def plot_magnification(self):
        """
        Plot the light curve.
        """
        # PLOT
        plt.plot(self.timeseries, self.magnification)
        plt.xlabel("Days")
        plt.ylabel("Magnification")
        plt.title(
            f"u0= {self.u0:3.5f}; tE= {self.tE:12.5f}; rho= {self.rho:8.5f};\n s= {self.s:3.5f}; "
            f"q= {self.q:8.5f}; alpha= {self.alpha:3.5f}"
        )
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
        t0 = lens_params["t0"]
        u0 = lens_params["u0"]
        einstein_crossing_time = lens_params["tE"]
        rho = lens_params["rho"]
        q = lens_params["q"]
        alpha = lens_params["alpha"]
        s = lens_params["s"]

        tau = (timeseries - t0) / einstein_crossing_time

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        x = tau * cos_alpha - u0 * sin_alpha
        y = tau * sin_alpha + u0 * cos_alpha

        # Conversion secondary body left -> right
        x = -x
        # Compute magnification
        accuracy = 1.0e-3  # Absolute mag accuracy (mag+/-accuracy)
        magnification = np.array([vbbmagU(s, q, rho, x[i], y[i], accuracy) for i in range(len(x))])
        return magnification
