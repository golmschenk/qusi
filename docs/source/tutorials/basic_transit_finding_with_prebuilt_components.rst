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

    python examples/download_spoc_transit_light_curves.py

The main thing to know is that this will create a `data` directory within the project directory and within that will be a `spoc_transit_experiment` directory, referring to the data for the experiment of finding transiting planets within the TESS SPOC data. This will further contain 3 directories. One for train data, one for validation data, and one for test data. Within each of those, it will create a `positive` directory, that will hold the light curves with transits, and a `negative` directory, that will hold the light curves without transits. So the project directory tree now looks like:

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

Each of these `positive` and `negative` data directories will now contain a set of light curves. The reason why the code in this script is not very important for you to know, is that it's mostly irrelevant for future uses. When you're working on your own problem, you'll obtain your data some other way. And `qusi` is flexible about the data structure, so this directory structure is not required. It's just one way to structure the data.

Train the network
-----------------

Next, we'll look at the `examples/transit_train.py` file. In this script is a `main` function which will train our neural network on our data. The training script has 3 main components:

#. Code to prepare our datasets.
#. Code to prepare the neural network model.
#. Code to running the training of the model on the datasets.

Since `qusi` provides both models and and training loop code, the only one of these components that every user will be expected to deal with is preparing the dataset, since you'll eventually want to have `qusi` tackle the task you're interested in which will require you're own data. And the `qusi` dataset component will help make your data more suitable for training a neural network. However, we're going to save how to set up your own dataset (and how these example datasets are created) for the next tutorial. For now, we'll just use the example datasets as is. So, in the example script, you will see the first couple of lines of the `main` function call other functions that produce an example train and validation dataset for us. Then we choose one of the neural network models `qusi` provides (in this case the `Hadryss` model). Then finally, we start the training session. To run this training, simply run the script with:

.. code:: sh

    python examples/transit_train.py

You should see some output showing basic training statistics from the terminal as it runs through the training loop. It will run for as many train cycles as were specified in the script.