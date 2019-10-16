.. image:: ramjet_engine.png

RAMjET RApid MachinE-learned Triage
====================================

**RAMjET is currently under development. This documentation is currently oriented toward the RAMjET development team.**

RAMjET is a framework for producing neural networks to characterize phenomena in astrophysical photometric data.

The basic idea of RAMjET is to take photometric data (either in the form of lightcurves or time-series of flux frames)
and automatically search for patterns in this data which correspond to astrophysical phenomena. The pipeline will then
automatically characterized these discovered patterns. These characterizations can come in several forms. The simplest
case would be classifying a lightcurve as containing a specific phenomena (e.g., this lightcurve contains a microlensing
event or not). However, the characterization could be more complex, such as providing a classification for each time
step (e.g., this time step contains a microlensing event) or even providing quantitative characterizations (e.g., this
time step is being magnified by a factor of X due to a microlensing event).

To produce such a neural network pipeline which can produce good predictions two main components need to be handled.
First, the data for training and testing the network needs to be prepared. Second, the network architecture along with
an appropriate training system needs to be designed. The main development focus of this project is provide a framework
to easily complete these two tasks, and to demonstrate the framework on specific examples.

.. toctree::

    input_data_types
    autoapi/index