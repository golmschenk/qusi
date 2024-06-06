# Basic transit identification dataset construction

This tutorial will show you how we built the datasets for the {doc}`/tutorials/basic_transit_identification_with_prebuilt_components` tutorial. This will also show how this can be adjusted for your own datasets. This tutorial expects you have already completed the {doc}`/tutorials/basic_transit_identification_with_prebuilt_components` tutorial and that you have the `examples` directory as obtained in that tutorial. Telling `qusi` how to access and use your data is the most complex part of using `qusi`. So if this tutorial seems a bit challenging, don't worry, this is hardest part.

## User defined components of a qusi dataset

For `qusi` to work with your data, it needs 3 components to be defined:

1. A function that finds your data files.
2. A function that loads the input data from a given file. The input data is often the fluxes and the times of a light curve.
3. A function that loads the label or target value for a file. This is what the neural network (NN) is going to try to predict, such as a classification label.

## Creating a function to find the data

The first thing we'll do is provide functions on where to find the data. A simple version of this might look like:

```python
def get_positive_train_paths():
    return list(Path('data/spoc_transit_experiment/train/positives').glob('*.fits'))
```

This functions says to create a `Path` object for a directory at `data/spoc_transit_experiment/train/positives`. Then, it obtains all the files ending with the `.fits` extension. It puts that in a list and returns that list. In particular, `qusi` expects a function that takes no input parameters and outputs a list of `Path`s.

In our example code, we've split the data based on if it's train, validation, or test data and we've split the data based on if it's positive or negative data. And we provide a function for each of the 6 permutations of this, which is almost identical to what's above. You can see the above function and other 5 similar functions near the top of `scripts/dataset.py`.

`qusi` is flexible in how the paths are provided, and this construction of having a separate function for each type of data is certainly not the only way of approaching this. Depending on your task, another option might serve better. In another tutorial, we will explore a few example alternatives. However, to better understand those alternatives, it's first useful to see the rest of this dataset construction.

## Creating a function to load the data

The next thing `qusi` needs is the function to load the data. In our example case, we use the same function to do this in all cases. It looks like:

```python
def load_times_and_fluxes_from_path(path):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes
```

This uses a builtin class in `qusi` that is designed for loading light curves from TESS mission FITS files. However, the important thing is that your function returns two comma separated values, which is a NumPy array of the times and a NumPy array of the fluxes of your light curve. And the function takes a single `Path` object as input. These `Path` objects will be one of the ones we returned from the functions in the previous section. But you can write any code you need to get from a `Path` to the two arrays that represent times and fluxes. For example, if your file is a simple CSV file, it would be easy to use Pandas to load the CSV file and extract the time column and the flux column as two arrays which are then returned at the end of the function. You will see the above function in `scripts/dataset.py`.

## Creating a function to provide a label for the data

Now we need to define what label belongs to each light curve in our data. In the next section, we're going to join the function we made previously to get the paths to the function that assigns the labels. Since we already defined different functions to get the paths for positive cases and negative cases and we're going to explicitly join this to a label function, for this current use case, we don't actually need the label function to contain any real logic in it. We're going to define two functions, that always return 0 (for negative) or always return 1 (for positive). They look like:

```python
def positive_label_function(path):
    return 1

def negative_label_function(path):
    return 0
```

Note, `qusi` expects the label functions to take in a `Path` object as input, even if we don't end up using it. This is because, it allows for more flexible configurations. For example, in a different situation, the data might not be split into positive and negative directories, but instead, the label data might be contained within the user's data file itself. Also, in other cases, this label can also be something other than 0 and 1. The label is whatever the NN is attempting to predict for the input light curve. But for our binary classification case, 0 and 1 are what we want to use. Once again, you can see these functions in `scripts/dataset.py`.

## Creating a light curve collection

Now we're going to join the various functions we've just defined into `LightCurveObservationCollection`s. For the case of positive train light curves, this looks like:

```python
positive_train_light_curve_collection = LightCurveObservationCollection.new(
    get_paths_function=get_positive_train_paths,
    load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
    load_label_from_path_function=positive_label_function)
```

This defines a collection of labeled light curves where `qusi` knows how to obtain the paths, how to load the times and fluxes of the light curves, and how to load the labels. This `LightCurveObservationCollection.new(...` function takes in the three pieces we just built earlier. Note that you pass in the functions themselves, not the output of the functions. So for the `get_paths_function` parameter, we pass `get_positive_train_paths`, not `get_positive_train_paths()` (notice the difference in parenthesis). `qusi` will call these functions internally. However, the above bit of code is not by itself in `scripts/dataset.py` as the rest of the code in this tutorial was. This is because `qusi` doesn't use this collection by itself. It uses it as part of a dataset. We will explain why there's this extra layer in a moment.

## Creating a dataset

Finally, we build the dataset `qusi` uses to train the network. First, we'll take a look and then unpack it:

```python
def get_transit_train_dataset():
    positive_train_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_positive_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_train_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_negative_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_train_light_curve_collection,
                                          negative_train_light_curve_collection])
    return train_light_curve_dataset
```

This is the function which generates the training dataset we called in the {doc}`/tutorials/basic_transit_identification_with_prebuilt_components` tutorial. The parts of this function are as follows. First, we create the `positive_train_light_curve_collection`. This is exactly what we just saw in the previous section. Next, we create a `negative_train_light_curve_collection`. This is almost identical to its positive counterpart, except now we pass the `get_negative_train_paths` and `negative_label_function` instead of the positive versions. Then there is the `train_light_curve_dataset = LightCurveDataset.new(` line. This creates a `qusi` dataset built from these two collections. The reason the collections are separate is that `LightCurveDataset` has several mechanisms working under-the-hood. Notably for this case, `LightCurveDataset` will balance the two light curve collections. We know of a lot more light curves that don't have planet transits in them than we do light curves that do have planet transits. In the real world case, it's thousands of times more at least. But for a NN, it's usually useful to during the training process to show equal amounts of the positives and negatives. `LightCurveDataset` will do this for us. You may have also noticed that we passed these collections in as the `standard_light_curve_collections` parameter. `LightCurveDataset` also allows for passing different types of collections. Notably, collections can be passed such that light curves from one collection will be injected into another. This is useful for injecting synthetic signals into real telescope data. However, we'll save the injection options for another tutorial.

You can see the above `get_transit_train_dataset` dataset creation function in the `scripts/dataset.py` file. The only part of that file we haven't yet looked at in detail is the `get_transit_validation_dataset` and `get_transit_finite_test_dataset` functions. However, these are nearly identical to the above `get_transit_train_dataset` expect using the validation and test path obtaining functions above instead of the train ones.

## Adjusting this for your own binary classification task

Now for a few quick starting points for how you might adjust this for your own binary classification task.

If you were simply looking for a different type of phenomena that wasn't transits, but were still searching in the TESS SPOC data, you would just need to replace the light curve FITS files in the data directory with the ones that match what you're trying to search for. Then, nothing else would need to change, and you would be able to use these same scripts to train a NN for search for that phenomena.

Let's say you weren't searching for TESS SPOC data, but were searching some other telescope data with light curves around the same length. If you're still able to put the light curves in the same data directory file structure as previously, you would just need to update the `load_times_and_fluxes_from_path` function to tell `qusi` how to get the times and fluxes from your file paths. And again, the existing scripts will allow you to start training a NN to search for your phenomena.
