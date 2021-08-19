from typing import Tuple, Optional
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils


class GaussianDropout(Layer):
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Arguments:
      rate: Float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

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
