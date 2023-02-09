Simple TESS data interface tutorial
===================================

Tutorial Summary
----------------

This tutorial shows how to use some convenience functions in the :code:`ramjet` package to quickly download,
view, and find additional information about `TESS <https://tess.mit.edu>`_ lightcurves.

Setup
-----
All that's needed for this tutorial is to install the :code:`ramjet` pip package using:

.. code-block:: bash

    pip install astroramjet

Downloading the light curve
---------------------------

To view a TESS light curve, we first need to get the light curve data from the Mikulski Archive for Space Telescopes
(MAST). To do this, we use the :code:`TessTwoMinuteCadenceLightCurve` class in the following way:

.. code-block:: python

    from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessTwoMinuteCadenceLightCurve
    light_curve = TessTwoMinuteCadenceLightCurve.from_mast(tic_id=370101492, sector=12)

Any valid TIC ID and sector can be used in the above code.

Viewing the light curve
-----------------------

Next, we'll get only the data points in the light curve where
the flux value is not NaN, because NaN values mess up the plotting of the light curve. To get these non-NaN values, we
run:

.. code-block:: python

    non_nan_flux_indexes = ~np.isnan(light_curve.fluxes)
    times = light_curve.times[non_nan_flux_indexes]
    fluxes = light_curve.fluxes[non_nan_flux_indexes]

Next we'll create and show the light curve figure:

.. code-block:: python

    figure = create_light_curve_figure(times=times,
                                   fluxes=fluxes,
                                   name='Relative flux',
                                   title=f'TIC {light_curve.tic_id} sector {light_curve.sector}')
    show(figure)

Checking for known planets
--------------------------

Checking for known planets requires a separate data interface. Running

.. code-block:: python

    from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface
    tess_toi_data_interface = TessToiDataInterface()
    tess_toi_data_interface.print_exofop_toi_and_ctoi_planet_dispositions_for_tic_target(tic_id=425997655)

will print any known `Exoplanet Follow-up Observing Program <https://exofop.ipac.caltech.edu/tess/>`_ planet
dispositions for the passed TIC ID. Note, this is not the same TIC ID we used above.

Checking for known variability
------------------------------

Running

.. code-block:: python

    tess_data_interface = TessDataInterface()
    tess_data_interface.print_variables_near_tess_target(tic_id=59661876)

will print any known stellar variability source near the TIC target. This information is based on the
`General Catalogue of Variable Stars <http://www.sai.msu.su/gcvs/gcvs/>`_. Note, this is not the same TIC ID we used
above.
