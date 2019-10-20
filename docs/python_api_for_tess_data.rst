Using the Python API for TESS data
==================================

If you just want to download all the TESS data related to the project, you can simply use the project's download
scripts. This page is to understand what those scripts are doing, and how to build your own script to download data for
a specific purpose.

There are several ways to access the TESS data, but the Python API for the data has several advantages, especially for
the RAMjET project. Unlike other forms of obtaining the data, such as from `TESS bulk download page
<http://archive.stsci.edu/tess/bulk_downloads.html>`_, a Python script can eliminate any human variability in the
download process and always produce the same result for each person. Because the download process is contained in a
single script, rather than a sequence of human run commands, it also means there's a record of exactly how the data
was obtained. Finally, the Python API gives much finer control of what data is downloaded. It can be pre-filtered before
downloading and can access earlier forms of data before the science product version.

Unfortunately, `the official documentation <https://astroquery.readthedocs.io/en/latest/mast/mast.html>`_ for obtaining
the TESS data with the Python API can seem a bit opaque to begin with. Here, we try to provide an understandable
walk-through to use the API.

Listing what targets have data
------------------------------

The first thing you'll need is the :code:`astroquery` Python package. Usually installed through :code:`pip`:

.. code-block:: bash

    pip install astroquery

This package provides the API to access the TESS data stored at the Mikulski Archive for Space Telescopes (MAST).

The first thing to do with this package is get the available :code:`Observation`'s (which is a class of the
:code:`astroquery.mast` package. What "observation" means here is a bit vague, but basically each :code:`Observation` is
a collection of related measurements from TESS. There are two main types of :code:`Observations` for TESS: time-series
and full frame images (FFIs). Time-series include the lightcurves and related transit detection information, as well as
the original target pixel files (TPFs) those lightcurves are generated from.

To get a single :code:`Observation` we can use something like:

.. code-block:: python

    from astroquery.mast import Observations
    observation = Observations.query_criteria(obs_collection='TESS', target_name='121420805')

This returns an :code:`astropy` :code:`Table` object where each row is an :code:`Observation`.
The :code:`query_criteria` command is used for all missions MAST stores data for. To specify we want TESS data, we are
filtering using :code:`obs_collection`. For TESS data, the :code:`target_name` refers to the TESS input catalog (TIC)
number of the target (note that it needs to be passed as a string). There are lots of other things we could filter on.
We can get the full list of filter parameters using :code:`Observations.get_metadata('observations')`. Be careful when
filtering though, as some of the parameter names are not particularly clear. For example, there is both a :code:`obs_id`
and a separate :code:`obsid` parameter which have very different values.

If instead of getting the just a single, we wanted to get all TESS observations, we could run:

.. code-block:: python

    observations = Observations.query_criteria(obs_collection='TESS')

This will give all TESS mission observations. We can then go through these observations to examine what data is
available for each.

With these observations we can use :code:`observation['dataproduct_type']` to view if the data is a time-series or full
frame image. If you are using PyCharm, you can convert the AstroPy :code:`Table` to a Pandas :code:`DataFrame` using
:code:`observation.to_pandas()`, then you can stop at a breakpoint to use the :code:`View as DataFrame` button in the
debugger "Variables" window to view all the table contents at once in an organized way.

Finding observation data products
---------------------------------

Assuming you've got an observation in a variable called :code:`observation` (or a bunch of them in a table), we can
find out what data is actually available for that observation. To do so, we can use:

.. code-block:: python

    product_list = Observations.get_product_list(observation)

These are data products which can be downloaded.
If you used the observation of TIC target 121420805 from above, you will get back a list of 8 entries (as of 20 October
2019). There's a TPF time-series in a FITS file, the lightcurve data in a FITS file, the data validation (DV) in a FITS
file, the DV in an XML file, and several PDFs describing the DV results in a human understandable
format.

First, an explanation of what the DV is. When the lightcurve is first produced, a Transit Planet Search (TPS) module
checks for threshold crossing events (things that might suggest a planet transit). If this turns up any sign of
anything, the lightcurve gets processed by the data validation (DV) pipeline. This process runs several fitting
algorithms to try to determine the properties of the candidate planet transit. The results of these algorithms are
stored in the DV files we see in the product list we just got from our code. From the product list, it's worth
downloading a couple of the PDFs to see what they look like. One PDF is a full summary of the DV and one is a one-page
summary. The remainder consist of one summary for each planet candidate, however, this information was already included
in the full summary.

The :code:`get_product_list` method can also be passed filters. The available filters can be listed using
:code:`Observations.get_metadata('products')`. And as before, you can explore the product table result (such as by
converting the AstroPy :code:`Table` to a Pandas :code:`DataFrame` and using PyCharm to view it as above).

Downloading the data products
-----------------------------

Finally, we can download the data products we've found. To do this, we take the AstroPy table of data products from
above and request the download:

.. code-block:: python

    manifest = Observations.download_products(product_list)

This will download all the files in the table. Note that this method does not return the downloaded data. Instead, it
returns a table explaining which data it downloaded and where it put it (hence the name "manifest"). Of course, you can
filter this product list before passing it to the download method. However, note that :code:`download_products` expects
a AstroPy :code:`Table`, not an individual :code:`Row` object.
