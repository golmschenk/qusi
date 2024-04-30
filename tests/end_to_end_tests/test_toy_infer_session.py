import os
from functools import partial

import numpy as np

from qusi.internal.infer_session import infer_session
from qusi.internal.device import get_device
from qusi.internal.light_curve_dataset import (
    default_light_curve_post_injection_transform,
)
from qusi.internal.single_dense_layer_model import SingleDenseLayerBinaryClassificationModel
from qusi.internal.toy_light_curve_collection import get_toy_finite_light_curve_dataset


def test_toy_infer_session():
    os.environ["WANDB_MODE"] = "disabled"
    model = SingleDenseLayerBinaryClassificationModel.new(input_size=100)
    test_light_curve_dataset = get_toy_finite_light_curve_dataset()
    test_light_curve_dataset.post_injection_transform = partial(
        default_light_curve_post_injection_transform, length=100
    )
    device = get_device()
    confidences = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                                batch_size=100, device=device)[0]
    assert isinstance(confidences, np.ndarray)
    assert 0 <= confidences[0] <= 1
