Basic transit finding with prebuilt components
==============================================

This tutorial will get you up and running with a neural network (NN) that can identify transiting exoplanets in data from the Transiting Exoplanet Survey Satellite (TESS). Many of the components used in this example will be prebuilt bits of code that we'll import from the package's example code. However, in later tutorials, we'll walkthrough how you would build each of these pieces yourself and how you would modify it for whatever your use case is.

Getting the example code
------------------------

First, create a directory to hold the project named `qusi_example_project`, or some other suitable name. Then get the example scripts from the `qusi` repository. You can download just that directory by clicking `here <https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2Fgolmschenk%2Fqusi%2Ftree%2Fmain%2Fexamples>`_. Move this `examples` directory into your project directory so that you have `qusi_example_project/examples`. The remainder of the commands will assume you are running code from the project directory, unless otherwise stated.

Downloading the dataset
-----------------------

The next thing we'll do is download a dataset of light curves that include cases both with and without transiting planets. To do this, run the example script at `examples/download_spoc_transit_light_curves`. For now, don't worry about how each part of the code works. You can run the script with:

.. code:: sh

    python examples/download_spoc_transit_light_curves

The main thing to know is that this will create a `data` directory within the project directory and within that will be a `spoc_transit_experiment` directory, referring to the data for the experiment of finding transiting planets within the TESS SPOC data. This will further contain 3 directories. One for train data, one for validation data, and one for test data. Within each of those, it will create a `positive` directory, that will hold the light curves with transits, and a `negative` directory, that will hold the light curves without transits. `qusi` is flexible about the data structure, so this format is not required for other problems, but it's one easy way to structure the data. So the project directory tree now looks like:

.. code::

    data
        spoc_transit_experiment
            test
                negative
                positive
            train
                negative
                positive
            validation
                negative
                positive
    examples
