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

Let's clarify what's happening here. The ``LightCurveDataset.new()`` constructor takes as input a parameter called `post_injection_transform`. This function will be applied to our light curves before they get handed to the NN. ``default_light_curve_post_injection_transform`` is the default set of preprocessing transforms ``qusi`` uses. We'll look at these transforms in more detail in the next section. ``partial`` is a Python builtin function, that takes another function as input, along with a parameter of that function, and creates a new function with that parameter prefilled. So ``partial(default_light_curve_post_injection_transform, length=4000)`` is taking our default transforms, setting the uniforming lengthening step to 4000, then giving us back the updated function, which we can then give to the dataset. The advantage to this approach is that ``post_injection_transform`` is completely flexible, as we'll explore more in the next section.

