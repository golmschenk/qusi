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
and full-frame images (FFIs).

To get the :code:`Observation`s for a single TESS input catalog (TIC) ID target we can use something like:

.. code-block:: python

    from astroquery.mast import Observations
    observations = Observations.query_criteria(obs_collection='TESS', target_name='25132999')

This returns an :code:`astropy` :code:`Table` object where each row is an :code:`Observation`.
The :code:`query_criteria` command is used for all missions MAST stores data for. To specify we want TESS data, we are
filtering using :code:`obs_collection`. For TESS data, the :code:`target_name` refers to the TIC
number of the target (note that it needs to be passed as a string). There are lots of other things we could filter on.
We can get the full list of filter parameters using :code:`Observations.get_metadata('observations')`. Be careful when
filtering though, as some of the parameter names are not particularly clear. For example, there is both a :code:`obs_id`
and a separate :code:`obsid` parameter which have very different values.

Looking at the results of this query, there are many :code:`Observation`s for this single target, all of which are of
the time-series type. Of the time-series type observations, 2 subcategory types exists for TESS: single-sector
observations and multi-sector observations. The :code:`dataURL` column of the :code:`Observation` will end in
:code:`lc.fits` for a single-sector observation (as the main data product is a lightcurve), and will end in
:code:`dvt.fits` for a multi-sector observation (as the main data product is a data validation time-series). For either,
the sector or sector range can be seen in the :code:`dataURL` or :code:`obs_id` column (e.g., `s0001-s0003`).

If instead of getting observations for a single target, we wanted to get all TESS observations, we could run:

.. code-block:: python

    observations = Observations.query_criteria(obs_collection='TESS')

This will give all TESS mission observations. We can then go through these observations to examine what data is
available for each.

With these observations we can use :code:`observations['dataproduct_type']` to view if the data is a time-series or full
frame image. If you are using PyCharm, you can convert the AstroPy :code:`Table` to a Pandas :code:`DataFrame` using
:code:`observations.to_pandas()`, then you can stop at a breakpoint to use the :code:`View as DataFrame` button in the
debugger "Variables" window to view all the table contents at once in an organized way.

Finding observation data products
---------------------------------

Assuming you've got an observation in a variable called :code:`observations` (or a bunch of them in a table), we can
find out what data is actually available for that observation. To do so, we can use:

.. code-block:: python

    product_list = Observations.get_product_list(observations)

These are data products which can be downloaded.
If you used an observation from the TIC target 25132999 from above, you will get back a list of entries.

If you passed an observation for a single-sector observation, you will get back a table containing a row for the
lightcurve data product and a row for the target pixel file (TPF) data product which that lightcurve was generated
from. For each of these lightcurves, a Transit Planet Search (TPS) module has been applied to search for threshold
crossing events (TCE), that is, things that might suggest a planet transit. If this turns up any sign of
anything, the lightcurve gets processed by the data validation (DV) pipeline. This process runs several fitting
algorithms to try to determine the properties of the candidate planet transit. The results of these algorithms are
stored in the DV files we see in the product list we just got from our code. From the product list, it's worth
downloading a couple of the PDFs to see what they look like. One PDF is a full summary of the DV and one is a one-page
summary. The remainder consists of one summary for each planet candidate, however, this information was already included
in the full summary. When listing the data products for this observation, these data products will be also be listed
if a TCE was triggered. If not, just the TPF and lightcurve data products will be listed.

If you tried to get the data products for a multi-sector observation, no lightcurves or TPFs will be listed. Instead,
you will get just the DV files for the range of sectors. These files are useful because they include the DV search
over multiple sectors, which gives the DV pipeline more lightcurve information to combine when searching for TCEs, but
you need to link this information back to the original lightcurves/TPFs from the other observations. Also note, the
observation list includes older multi-sector DV runs. That is, if the target was included in sectors 1 - 5, a DV run
may have earlier been performed for sectors 1 - 3. However, a newer DV run which includes all sectors 1 - 5 might now
exist, and the older 1 - 3 one is probably obsolete.

A full description of all the data products of TESS can be found in the `TESS Science Data Products Description
Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf>`_.

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
an AstroPy :code:`Table`, not an individual :code:`Row` object.
