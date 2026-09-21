import pytest

from qusi.internal.training_hyperparameter_configuration import TrainingHyperparameterConfiguration


def test_hyperparameter_configuration_requires_a_batch_size():
    with pytest.raises(ValueError):
        TrainingHyperparameterConfiguration.new(global_batch_size=None, local_batch_size=None)

def test_hyperparameter_configuration_one_set_batch_size_set_does_not_raise():
    TrainingHyperparameterConfiguration.new(global_batch_size=1, local_batch_size=None)
    TrainingHyperparameterConfiguration.new(global_batch_size=None, local_batch_size=1)

def test_hyperparameter_configuration_with_both_batch_sizes_set_raises():
    with pytest.raises(ValueError):
        TrainingHyperparameterConfiguration.new(global_batch_size=1, local_batch_size=1)

def test_setting_of_batch_sizes_based_on_world_size():
    configuration0 = TrainingHyperparameterConfiguration.new(global_batch_size=70, local_batch_size=None)
    configuration0.update_batch_sizes_based_on_world_size(7)
    assert configuration0.local_batch_size == 10

    configuration1 = TrainingHyperparameterConfiguration.new(global_batch_size=None, local_batch_size=10)
    configuration1.update_batch_sizes_based_on_world_size(7)
    assert configuration1.global_batch_size == 70

    # Test rounding.
    configuration2 = TrainingHyperparameterConfiguration.new(global_batch_size=71, local_batch_size=None)
    configuration2.update_batch_sizes_based_on_world_size(7)
    assert configuration2.local_batch_size == 10

    # Test minimal local batch size is 1.
    configuration2 = TrainingHyperparameterConfiguration.new(global_batch_size=1, local_batch_size=None)
    configuration2.update_batch_sizes_based_on_world_size(7)
    assert configuration2.local_batch_size == 1
