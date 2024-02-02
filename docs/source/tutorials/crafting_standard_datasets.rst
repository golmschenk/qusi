Crafting standard datasets
==========================

In the earlier tutorials, we saw an example of how to use ``qusi`` on a transit dataset and how that dataset was constructed. We'll expand on the ways we can work with the datasets here to create a dataset that allows you to apply ``qusi`` to your own dataset.

The content of this tutorial will round out and material you need to create a standard binary or multi-class classification problem with simple light curves. In addition to the content of this tutorial, ``qusi`` includes a way easily to inject data into light curves (with any mix of real or synthetic data). This will be discussed in :doc:`/tutorials/crafting_injected_datasets`. Finally, ``qusi`` is not limited to simple light curves and classification. We'll explore more complex datasets in :doc:`/tutorials/advanced_crafting_datasets`.

Light curve lengths
-------------------

In the :doc:`/tutorials/basic_transit_identification_dataset_construction` tutorial, we showed how the transit dataset was constructed. One piece we (purposely) glossed over at the time, is that ``qusi`` is doing several preprocessing steps on the data before it's handed to the neural network (NN). We'll talk more about the other preprocessing steps in the next section, but the first one to note is a uniform lengthening step. Some of the NNs used by ``qusi`` (though not all) require a uniform light curve input length. To accomplish this, the default preprocessing will truncate light curves that are too long and repeat light curves that are too short. Perhaps surprisingly, in addition to enabling NNs that require uniform length, this uniform lengthening (along with some other preprocessing) can actually improve the NN's predictive performance by reducing overfitting.

However, the uniform length is set to a specific default value. A good choice for this might be the median length of the light curves in your dataset. For now, we'll set the length for the preprocessing step, but keep all other preprocessing steps the same. To do this, at the top of our dataset file, we'll first import some necessary functions:

.. code:: python

    from functools import partial
    from qusi.light_curve_dataset import default_light_curve_post_injection_transform

Then, were we specify the construction of our dataset, we'll add an additional input parameter. So taking what we had in the previous tutorial, we can now change the dataset creation statement to:

.. code:: python

    train_light_curve_dataset = LightCurveDataset.new(
            standard_light_curve_collections=[positive_train_light_curve_collection,
                                              negative_train_light_curve_collection]
            post_injection_transform=partial(default_light_curve_post_injection_transform, length=4000)
    )

Let's clarify what's happening here. The ``LightCurveDataset.new()`` constructor takes as input a parameter called ``post_injection_transform``. This function will be applied to our light curves before they get handed to the NN. ``default_light_curve_post_injection_transform`` is the default set of preprocessing transforms ``qusi`` uses. We'll look at these transforms in more detail in the next section. ``partial`` is a Python builtin function, that takes another function as input, along with a parameter of that function, and creates a new function with that parameter prefilled. So ``partial(default_light_curve_post_injection_transform, length=4000)`` is taking our default transforms, setting the uniforming lengthening step to 4000, then giving us back the updated function, which we can then give to the dataset. The advantage to this approach is that ``post_injection_transform`` is completely flexible, as we'll explore more in the next section.

Before we run the updated code, we also need to use a NN model which expects our new input size. Luckily, ``qusi`` has NN architectures that automatically resize their components for a given input size. So the only other change from the existing code is to change ``Hadryss.new()`` to ``Hadryss.new(input_length=4000)``.

Modifying the preprocessing
---------------------------

In the previous section, we only changed the length of that the uniform lengthening preprocessing transform is using. However, we still used all the remaining default preprocessing steps that are contained in ``default_light_curve_post_injection_transform``. Let's take a look at what the default one does. It looks like:

.. code:: python

    def default_light_curve_observation_post_injection_transform(x: LightCurveObservation, length: int) -> (Tensor, Tensor):
        x = remove_nan_flux_data_points_from_light_curve_observation(x)
        x = randomly_roll_light_curve_observation(x)
        x = from_light_curve_observation_to_fluxes_array_and_label_array(x)
        x = make_fluxes_and_label_array_uniform_length(x, length=length)
        x = pair_array_to_tensor(x)
        x = (normalize_tensor_by_modified_z_score(x[0]), x[1])
        return x

It's a function that takes in a ``LightCurveObservation`` and spits out two ``Tensor``\s, one for the fluxes and one for the label to predict. Most of the data transform functions within have names that are largely descriptive, but we'll walk through them anyway. ``remove_nan_flux_data_points_from_light_curve_observation`` removes time steps from a ``LightCurveObservation`` where the flux is NaN. ``randomly_roll_light_curve_observation`` randomly rolls the light curve (a random cut is made and the two segments' order is swapped). ``from_light_curve_observation_to_fluxes_array_and_label_array`` extracts two NumPy arrays from a ``LightCurveObservation``, one for the fluxes and one from the label (which in this case will be an array with a single value). ``make_fluxes_and_label_array_uniform_length`` performs the uniform lengthening we discussed in the previous section. ``pair_array_to_tensor`` converts the pair of NumPy arrays to a pair of PyTorch tensors (PyTorch's equivalent of an array). ``normalize_tensor_by_modified_z_score`` normalizes a tensor via based on the median absolute deviation. Notice, this is only applied to the flux tensor, not the label tensor.

It's worth noting, ``default_light_curve_post_injection_transform`` is just a function that can be replaced as desired. To remove one of the preprocessing steps or add in an addition one, we can simply make a modified version of this function. Additionally, ``qusi`` does not require the transform function to output only the fluxes and a binary label. The ``Hadryss`` NN model expects these two types of values for training, but other models may take advantage of the times of the light curve, or they may predict multi-class or regression labels.