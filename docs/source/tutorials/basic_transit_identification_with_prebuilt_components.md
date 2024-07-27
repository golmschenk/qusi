# Basic transit identification with prebuilt components

This tutorial will get you up and running with a neural network (NN) that can identify transiting exoplanets in data from the Transiting Exoplanet Survey Satellite (TESS). Many of the components used in this example will be prebuilt bits of code that we'll import from the package's example code. However, in later tutorials, we'll walk through how you would build each of these pieces yourself and how you would modify it for whatever your use case is.

## Getting the example code

First, we'll download some example code and enter that project's directory. To do this, run
```sh
git clone https://github.com/golmschenk/qusi_example_transit_binary_classification.git
cd qusi_example_transit_binary_classification
```
The remainder of the commands will assume you are running code from the project directory, unless otherwise stated.

## Downloading the dataset

The next thing we'll do is download a dataset of light curves that include cases both with and without transiting planets. To do this, run the example script at `scripts/download_data.py`. For now, don't worry about how each part of the code works. You can run the script with

```sh
python scripts/download_data.py
```

The main thing to know is that this will create a `data` directory within the project directory and within that will be a `spoc_transit_experiment` directory, referring to the data for the experiment of finding transiting planets within the TESS SPOC data. This will further contain 3 directories. One for train data, one for validation data, and one for test data. Within each of those, it will create a `positive` directory, that will hold the light curves with transits, and a `negative` directory, that will hold the light curves without transits. So the project directory tree now contains

```
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
        infer
```

Each of these `positive` and `negative` data directories will now contain a set of light curves. The reason why the code in this script is not very important for you to know, is that it's mostly irrelevant for future uses. When you're working on your own problem, you'll obtain your data some other way. And `qusi` is flexible about the data structure, so this directory structure is not required. It's just one way to structure the data. Note, this is a relatively small dataset to make sure it doesn't take very long to get up and running. To get a better result, you'd want to download all known transiting light curves and a much larger collection non-transiting light curves. To quickly visualize one of these light curves, you can use the script at `scripts/light_curve_visualization.py`. Due to the available light curves on MAST being updated constantly, the random selection of light curves you downloaded might not include the light curve noted in this example file. Be sure to open the `scripts/light_curve_visualization.py` file and update the path to one of the light curves you downloaded. To see a transit case, be sure to select one from one of the `positive` directories. Then run

```sh
python scripts/light_curve_visualization.py
```

You should see something like

```{image} light_curve_example.png
```

where, in this case, the temporary dips are transiting events.

## Preparing for training

`qusi` uses Weights & Biases (`wandb`), a machine learning logging platform, to record metrics from training experiments. Among other things, it will create plots showing the training progress and allow easy comparison among the various runs. While you can run the `wandb` platform locally, it's easiest to use their cloud platform, which has [free academic research team projects and free personal projects](https://wandb.ai/site/pricing). To use it with `qusi`, [sign up for an account](https://wandb.ai/site), then from your project directory use

```sh
wandb login
```

to login. If you want to proceed without a `wandb` account and log the data offline, you will need to run

```sh
(cd sessions && wandb offline)
```

This will only log runs locally. If you choose the offline route, at some point, you will want to follow [their guide to run the local server](https://docs.wandb.ai/guides/hosting/how-to-guides/basic-setup) so that you can view the metric plots. However, for the moment, just running `wandb offline` (from the sessions directory) will allow you to proceed with this tutorial.

## Train the network

Next, we'll look at the `scripts/train.py` file. In this script is a `main` function which will train our neural network on our data. The training script has 3 main components:

1. Code to prepare our datasets.
2. Code to prepare the neural network model.
3. Code to running the training of the model on the datasets.

Since `qusi` provides both models and and training loop code, the only one of these components that every user will be expected to deal with is preparing the dataset, since you'll eventually want to have `qusi` tackle the task you're interested in which will require you're own data. And the `qusi` dataset component will help make your data more suitable for training a neural network. However, we're going to save how to set up your own dataset (and how these example datasets are created) for the next tutorial. For now, we'll just use the example datasets as is. So, in the example script, you will see the first couple of lines of the `main` function call other functions that produce an example train and validation dataset for us. Then we choose one of the neural network models `qusi` provides (in this case the `Hadryss` model). Then finally, we start the training session. To run this training, simply run the script with:

```sh
python scripts/train.py
```

You should see some output showing basic training statistics from the terminal as it runs through the training loop. It will run for as many train cycles as were specified in the script. On every completed cycle, `qusi` will save the latest version of the fitted model to `sessions/<wandb_run_name>/latest_model`.

You can also go to your Wandb project to see the metrics over the course of the training in plot form.

## Test the fitted model

A "fitted model" is a model which has been trained, or fitted, on some training data. Next, we'll take the fitted model we produced during training, and test it on data it didn't see during the training process. This is what happens in the `scripts/finite_dataset_test.py` script. The `main` function will look semi-similar to from the training script. Again, we'll defer how the dataset is produced until the next tutorial. Then we create the model as we did before, but this time we load the fitted parameters of the model from the saved file. Here, you will need to update the script to point to your saved model produced in the last section. Then we can run the script with

```sh
python scripts/finite_dataset_test.py
```

This will run the network on the test data, producing the metrics that are requested in the file.
