import os
from functools import partial

from qusi.internal.light_curve_dataset import (
    default_light_curve_observation_post_injection_transform,
)
from qusi.internal.single_dense_layer_model import SingleDenseLayerBinaryClassificationModel
from qusi.internal.toy_light_curve_collection import get_toy_dataset
from qusi.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from qusi.internal.train_session import train_session


def test_toy_train_session():
    os.environ["WANDB_MODE"] = "disabled"
    model = SingleDenseLayerBinaryClassificationModel.new(input_size=100)
    dataset = get_toy_dataset()
    dataset.post_injection_transform = partial(
        default_light_curve_observation_post_injection_transform, length=100
    )
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=3, cycles=2, train_steps_per_cycle=5, validation_steps_per_cycle=5
    )
    train_session(
        train_datasets=[dataset],
        validation_datasets=[dataset],
        model=model,
        hyperparameter_configuration=train_hyperparameter_configuration,
    )
