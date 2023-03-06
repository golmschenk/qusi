"""
Code for a general convolutional model for light curve data.
"""
from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, Concatenate, RepeatVector

from ramjet.models.components.light_curve_network_block import LightCurveNetworkBlock


class Hades(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=3)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False, dropout_rate=0.5)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=8, pooling_size=1, dropout_rate=0.5)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class HadesWithoutBatchNormalization(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=3, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0.5)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0.5)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0.5, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=8, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class HadesWithAuxiliaryNoSigmoid(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=3)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False, dropout_rate=0.5)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.repeat_auxiliary_values_layer = RepeatVector(35)
        self.concatenate = Concatenate()
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        light_curves, auxiliary_informations = inputs
        x = light_curves
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        aux = self.repeat_auxiliary_values_layer(auxiliary_informations)
        x = self.concatenate([x, aux])
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class HadesNoSigmoid(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=3)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False, dropout_rate=0.5)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class FfiHades(Model):
    def __init__(self):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2)
        self.block2 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2)
        self.block3 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(1, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.
        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class HadesWithoutBatchNormalization(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=3, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5,
                                             batch_normalization=False)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5,
                                             batch_normalization=False)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False, dropout_rate=0.5,
                                             batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5,
                                             batch_normalization=False)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class HadesSmallWithoutBatchNormalization(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block7 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block8 = LightCurveNetworkBlock(filters=10, kernel_size=3, pooling_size=1, spatial=False, batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=10, kernel_size=7, pooling_size=1, batch_normalization=False)
        self.block10 = LightCurveNetworkBlock(filters=10, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(1, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.
        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class HadesSmallWideEndWithoutBatchNormalization(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block7 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block8 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=1, spatial=False, batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=100, kernel_size=7, pooling_size=1, batch_normalization=False)
        self.block10 = LightCurveNetworkBlock(filters=100, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(1, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.
        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class HadesLargeWithoutBatchNormalization(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=256, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=256, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block7 = LightCurveNetworkBlock(filters=256, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block8 = LightCurveNetworkBlock(filters=200, kernel_size=3, pooling_size=1, spatial=False, batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=200, kernel_size=7, pooling_size=1, batch_normalization=False)
        self.block10 = LightCurveNetworkBlock(filters=200, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(1, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.
        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs


class HadesRegularResizedForFfi(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout_rate=0.5)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout_rate=0.5)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1, spatial=False, dropout_rate=0.5)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class HadesRegularResizedForFfiNoBatchNormalization(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout_rate=0.5, batch_normalization=False)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout_rate=0.5, batch_normalization=False)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1, spatial=False, dropout_rate=0.5, batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5, batch_normalization=False)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class HadesRegularResizedForFfiNoDropout(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout_rate=0)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=1, dropout_rate=0)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1, spatial=False, dropout_rate=0)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class Hades18000(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=3)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False, dropout_rate=0.5)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=8, pooling_size=1, dropout_rate=0.5)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs