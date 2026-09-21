import os
from functools import partial

from qusi.internal.light_curve_dataset import (
    default_light_curve_observation_post_injection_transform,
)
from qusi.internal.single_dense_layer_model import SingleDenseLayerBinaryClassificationModel
from qusi.internal.toy_light_curve_collection import get_toy_dataset
from qusi.internal.training_hyperparameter_configuration import TrainingHyperparameterConfiguration
from qusi.internal.lightning_train_session import run_training_session
from qusi.internal.training_system_configuration import TrainingSystemConfiguration


def test_toy_train_session():
    os.environ["WANDB_MODE"] = "disabled"
    model = SingleDenseLayerBinaryClassificationModel.new(input_size=100)
    dataset = get_toy_dataset()
    dataset.post_injection_transform = partial(
        default_light_curve_observation_post_injection_transform, length=100
    )
    train_hyperparameter_configuration = TrainingHyperparameterConfiguration.new(
        global_batch_size=3, cycles=2, train_steps_per_cycle=5, validation_steps_per_cycle=5
    )
    train_system_configuration = TrainingSystemConfiguration.new(accelerator='cpu', data_workers_per_train_process=1)
    run_training_session(
        training_datasets=[dataset],
        validation_datasets=[dataset],
        model=model,
        hyperparameter_configuration=train_hyperparameter_configuration,
        system_configuration=train_system_configuration,
    )
