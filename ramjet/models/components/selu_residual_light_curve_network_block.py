"""
Code for a residual light curve network block.
"""
from typing import Optional

from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization,\
    Layer, Permute, ZeroPadding1D, AveragePooling1D, AlphaDropout
from tensorflow.keras.activations import selu
from tensorflow.keras.initializers import LecunNormal


class SeluResidualLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        dimension_decrease_factor = 4
        kernel_initializer = LecunNormal()
        self.dimension_decrease_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=selu,
            kernel_initializer=kernel_initializer)
        self.convolutional_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=selu,
            padding='same', kernel_initializer=kernel_initializer)
        self.dimension_increase_layer = Convolution1D(
            output_channels, kernel_size=1, activation=selu, kernel_initializer=kernel_initializer)
        if pooling_size > 1:
            self.pooling_layer = AveragePooling1D(pool_size=pooling_size, padding='same')
        else:
            self.pooling_layer = None
        if input_channels is not None and output_channels != input_channels:
            if output_channels < input_channels:
                raise NotImplementedError(f'Residual blocks with less output channels than input channels is not'
                                          f'implemented. Output channels was {output_channels} and input was'
                                          f'{input_channels}')
            self.dimension_change_permute0 = Permute((2, 1))
            self.dimension_change_layer = ZeroPadding1D(padding=(0, output_channels - input_channels))
            self.dimension_change_permute1 = Permute((2, 1))
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = AlphaDropout(rate=dropout_rate, noise_shape=(50, 1, output_channels))
        else:
            self.dropout_layer = None

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the block.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        y = x
        y = self.dimension_decrease_layer(y, training=training)
        y = self.convolutional_layer(y, training=training)
        y = self.dimension_increase_layer(y, training=training)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x, training=training)
            y = self.pooling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        return x + y
