"""
This is a model to use Naoki's genstars
"""

import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd


def get_header(command):
    ROMAN = 0
    VERBOSITY = 0
    iMag = -1
    args = command.split()
    for i, arg in enumerate(args):
        if arg == 'ROMAN':
            ROMAN = int(args[i + 1])
        if arg == 'VERBOSITY':
            VERBOSITY = int(args[i + 1])
        if arg == 'iMag':
            iMag = int(args[i + 1])
    if iMag == -1:
        if (ROMAN == 1):
            iMag = 4
        else:
            iMag = 3

    if ROMAN == 1:
        Mags = ['J', 'H', 'Ks', 'Z087', 'W146', 'F213']
    else:
        Mags = ['V', 'I', 'J', 'H', 'Ks']

    if VERBOSITY == 3:
        Alams = ['A' + mag for mag in Mags]
    else:
        Alams = ['A' + Mags[iMag]]

    Mags = [mag + '-mag' for mag in Mags]

    if VERBOSITY == 3:
        header = Mags + Alams + 'Mass      Radius   Distance      mu_l      mu_b   l   b cls fREM InitialMass      v_x      v_y      v_z'.split()
    elif VERBOSITY == 2:
        header = Mags + 'Mass      Radius   Distance      mu_l      mu_b'.split() + Alams + 'l  b cls fREM   InitialMass      v_x      v_y      v_z'.split()
    elif VERBOSITY == 1:
        header = Mags + 'Mass      Radius   Distance      mu_l      mu_b'.split() + Alams + 'l  b cls fREM'.split()
    else:
        header = None

    return header


def run_genstars(longitude_l_interval_, latitude_b_interval_, is_it_roman_, fraction_output_size_, iMag_band_,
                 Mag_range_, verbosity_, random_seed_, outfile_):

    longitude_l_min,  longitude_l_max = longitude_l_interval_
    latitude_b_min,  latitude_b_max = latitude_b_interval_
    Mag_min, Mag_max = Mag_range_
    command = f"./genstars l {longitude_l_min} {longitude_l_max} b {latitude_b_min} {latitude_b_max} " \
              f"ROMAN {is_it_roman_} fSIMU {fraction_output_size_} iMag {iMag_band_} Magrange {Mag_min} {Mag_max} " \
              f"VERBOSITY {verbosity_} seed {random_seed_} > {outfile_}"
    print("{}".format(command))
    subprocess.run(command, shell=True, text=True)
    return get_header(command)

if __name__ == '__main__':
    # Conduct genstars and output the star (13 < H < 20) catalog to tmp.dat
    longitude_l_interval = [0.9, 1.1]
    latitude_b_interval = [-0.8, -0.6]
    is_it_roman = 0 # Roman = 1
    fraction_output_size = 0.01
    # No Red band
    # prime 3 = H
    # roman 1 = H
    iMag_band = 3
    Mag_range = [13, 20]
    verbosity = 1
    random_seed = 2

    outfile = '../ramjet/photometric_database/list_of_simulated_stars_prime.dat'

    run_genstars(longitude_l_interval, latitude_b_interval, is_it_roman, fraction_output_size, iMag_band,
                 Mag_range, verbosity, random_seed, outfile)


    longitude_l_interval = [0.9, 1.1]
    latitude_b_interval = [-0.8, -0.6]
    is_it_roman = 1 # Roman = 1
    fraction_output_size = 0.01
    # No Red band
    # prime 3 = H
    # roman 1 = H
    iMag_band = 1
    Mag_range = [13, 20]
    verbosity = 1
    random_seed = 2

    outfile = '../ramjet/photometric_database/list_of_simulated_stars_roman.dat'

    run_genstars(longitude_l_interval, latitude_b_interval, is_it_roman, fraction_output_size, iMag_band,
                 Mag_range, verbosity, random_seed, outfile)

