from typing import Tuple, Optional
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils


class GaussianDropout(Layer):
    def __init__(self, rate, noise_shape: Optional[Tuple[int, ...]] = None, **kwargs):
        super(GaussianDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.noise_shape = noise_shape

    def call(self, inputs, training=None):
        if 0 < self.rate < 1:
            if self.noise_shape is not None:
                noise_shape = self.noise_shape
            else:
                noise_shape = array_ops.shape(inputs)

            def noised():
                stddev = np.sqrt(self.rate / (1.0 - self.rate))
                return inputs * K.random_normal(
                    shape=noise_shape,
                    mean=1.0,
                    stddev=stddev,
                    dtype=inputs.dtype)

            return K.in_train_phase(noised, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(GaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
