"""
Code for a residual light curve network block.
"""
import math
from typing import Optional

import tensorflow
from tensorflow.keras.layers import LeakyReLU, Convolution1D, MaxPooling1D, BatchNormalization,\
    Layer, Permute, ZeroPadding1D, SpatialDropout1D
from tensorflow.keras.regularizers import L2
from tensorflow.python.keras.layers import Cropping1D


class ResidualLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
            self.batch_normalization1 = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.convolutional_layer0 = Convolution1D(
            output_channels, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.convolutional_layer1 = Convolution1D(
            output_channels, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.pooling_layer = MaxPooling1D(pool_size=pooling_size, padding='same')
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
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
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
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
        y = self.convolutional_layer0(y, training=training)
        if self.batch_normalization is not None:
            y = self.batch_normalization1(y, training=training)
        y = self.convolutional_layer1(y, training=training)
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


class BottleNeckResidualLightCurveNetworkBlock(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dimension_decrease_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolutional_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.dimension_increase_layer = Convolution1D(
            output_channels, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.pooling_layer = MaxPooling1D(pool_size=pooling_size, padding='same')
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
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
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
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
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

class BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dimension_decrease_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolutional_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.dimension_increase_layer = Convolution1D(
            output_channels, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.pooling_layer = MaxPooling1D(pool_size=pooling_size, padding='same')
        else:
            self.pooling_layer = None
        if input_channels is not None and output_channels != input_channels:
            if output_channels < input_channels:
                raise NotImplementedError(f'Residual blocks with less output channels than input channels is not'
                                          f'implemented. Output channels was {output_channels} and input was'
                                          f'{input_channels}')
            self.repeats = math.ceil(output_channels / input_channels)
            cropping_amount = (input_channels * self.repeats) - output_channels
            self.dimension_change_permute0 = Permute((2, 1))
            self.dimension_change_layer = Cropping1D((0, cropping_amount))
            self.dimension_change_permute1 = Permute((2, 1))
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
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
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
        y = self.dimension_decrease_layer(y, training=training)
        y = self.convolutional_layer(y, training=training)
        y = self.dimension_increase_layer(y, training=training)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x, training=training)
            y = self.pooling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = tensorflow.repeat(x, self.repeats, axis=2)
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        z = x + y
        return z


class BottleNeckResidualLightCurveNetworkBlockMainPathRepeatMainPathDropout(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dimension_decrease_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolutional_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.dimension_increase_layer = Convolution1D(
            output_channels, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.pooling_layer = MaxPooling1D(pool_size=pooling_size, padding='same')
        else:
            self.pooling_layer = None
        if input_channels is not None and output_channels != input_channels:
            if output_channels < input_channels:
                raise NotImplementedError(f'Residual blocks with less output channels than input channels is not'
                                          f'implemented. Output channels was {output_channels} and input was'
                                          f'{input_channels}')
            self.repeats = math.ceil(output_channels / input_channels)
            cropping_amount = (input_channels * self.repeats) - output_channels
            self.dimension_change_permute0 = Permute((2, 1))
            self.dimension_change_layer = Cropping1D((0, cropping_amount))
            self.dimension_change_permute1 = Permute((2, 1))
        else:
            self.dimension_change_layer = None
        if dropout_rate > 0:
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
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
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
        y = self.dimension_decrease_layer(y, training=training)
        y = self.convolutional_layer(y, training=training)
        y = self.dimension_increase_layer(y, training=training)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x, training=training)
            y = self.pooling_layer(y, training=training)
        if self.dimension_change_layer is not None:
            x = tensorflow.repeat(x, self.repeats, axis=2)
            x = self.dimension_change_permute0(x, training=training)
            x = self.dimension_change_layer(x, training=training)
            x = self.dimension_change_permute1(x, training=training)
        z = x + y
        if self.dropout_layer is not None:
            z = self.dropout_layer(z, training=training)
        return z

class BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
            self.batch_normalization1 = BatchNormalization(scale=False)
            self.batch_normalization2 = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dimension_decrease_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolutional_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.dimension_increase_layer = Convolution1D(
            output_channels, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.pooling_layer = MaxPooling1D(pool_size=pooling_size, padding='same')
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
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
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
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
        y = self.dimension_decrease_layer(y, training=training)
        if self.batch_normalization is not None:
            y = self.batch_normalization1(y, training=training)
        y = self.convolutional_layer(y, training=training)
        if self.batch_normalization is not None:
            y = self.batch_normalization2(y, training=training)
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


class BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNormBnAfterActivations(Layer):
    def __init__(self, output_channels: int, input_channels: Optional[int] = None, kernel_size: int = 3,
                 pooling_size: int = 1, batch_normalization: bool = True, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        dimension_decrease_factor = 4
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
            self.batch_normalization1 = BatchNormalization(scale=False)
            self.batch_normalization2 = BatchNormalization(scale=False)
        else:
            self.batch_normalization = None
        if l2_regularization > 0:
            l2_regularizer = L2(l2_regularization)
        else:
            l2_regularizer = None
        self.dimension_decrease_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolutional_layer = Convolution1D(
            output_channels // dimension_decrease_factor, kernel_size=kernel_size, activation=leaky_relu,
            padding='same', kernel_regularizer=l2_regularizer)
        self.dimension_increase_layer = Convolution1D(
            output_channels, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        if pooling_size > 1:
            self.pooling_layer = MaxPooling1D(pool_size=pooling_size, padding='same')
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
            self.dropout_layer = SpatialDropout1D(rate=dropout_rate)
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
        if self.batch_normalization is not None:
            y = self.batch_normalization(y, training=training)
        y = self.convolutional_layer(y, training=training)
        if self.batch_normalization is not None:
            y = self.batch_normalization1(y, training=training)
        y = self.dimension_increase_layer(y, training=training)
        if self.batch_normalization is not None:
            y = self.batch_normalization2(y, training=training)
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