Basic transit identification dataset construction
=================================================

This tutorial will show you how we built the datasets for the :doc:`/tutorials/basic_transit_identification_with_prebuilt_components` tutorial. This will also show how this can be adjusted for your own datasets. This tutorial expects you have already completed the :doc:`/tutorials/basic_transit_identification_with_prebuilt_components` tutorial and that you have the `examples` directory as obtained in that tutorial.

User defined components of a qusi dataset
-----------------------------------------

For ``qusi`` to work with your data, it needs 3 components to be defined:

1. A function that finds your data files.
2. A function that loads the input data from a given file. The input data is often the fluxes and the times of a light curve.
3. A function that loads the label or target value for a file. This is what the neural network is going to try to predict, such as a classification label.

Finding the data
----------------

The first thing we'll do is provide functions on where to find the data. A simple version of this might look like:

.. code:: python

    def get_positive_train_paths():
        return list(Path('data/spoc_transit_experiment/train/positives').glob('*.fits'))

This functions says to create a ``Path`` object for a directory at ``data/spoc_transit_experiment/train/positives``. Then, it obtains all the files ending with the ``.fits`` extension. It puts that in a list and returns that list. In particular, ``qusi`` expects a function that takes no input parameters and outputs a list of ``Path``\s.

In our example code, we've split the data based on if it's train, validation, or test data and we've split the data based on if it's positive or negative data. And we provide a function for each of the 6 permutations of this, which is almost identical to what's above. You can see the above function and other 5 similar functions near the top of ``examples/transit_dataset.py``.

``qusi`` is flexible in how the paths are provided, and this construction of having a separate function for each type of data is certainly not the only way of approaching this. Depending on your task, another option might serve better. Later, we will explore a few example alternatives. However, to better understand those alternatives, it's first useful to see the rest of this dataset construction.