"""
Code for a general convolutional model for light curve data.
"""
from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D

from ramjet.models.components.light_curve_network_block import LightCurveNetworkBlock


class GmlModel(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        self.block0 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=64, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=64, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=64, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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


class GmlModel1(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        self.block0 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=64, kernel_size=4, pooling_size=4, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=64, kernel_size=4, pooling_size=4, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=64, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block9 = LightCurveNetworkBlock(filters=20, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block10 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0, l2_regularization=l2_regularization)
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


class GmlModel2(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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


class GmlModel2Wider(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        self.block0 = LightCurveNetworkBlock(filters=16*2, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32*2, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64*2, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128*2, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128*2, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128*2, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128*2, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128*2, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40*2, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40*2, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40*2, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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

class GmlModel2LessBatchNorm(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, batch_normalization=False)
        self.block4 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization, batch_normalization=False)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False, batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=40, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False, batch_normalization=False)
        self.block10 = LightCurveNetworkBlock(filters=40, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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

class GmlModel2NoL2(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.0
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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


class GmlModel2WiderNoL2(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.0
        wider_factor = 2
        self.block0 = LightCurveNetworkBlock(filters=16*wider_factor, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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


class GmlModel2Wider4NoL2(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.0
        wider_factor = 4
        self.block0 = LightCurveNetworkBlock(filters=16*wider_factor, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=0.5, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=7, pooling_size=1, dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=0.5, l2_regularization=l2_regularization, spatial=False)
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

class GmlModel2Wider4NoL2NoDo(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.0
        wider_factor = 4
        dropout_rate = 0.0
        self.block0 = LightCurveNetworkBlock(filters=16*wider_factor, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=7, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
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

class GmlModel2Wider4(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        wider_factor = 4
        dropout_rate = 0.5
        self.block0 = LightCurveNetworkBlock(filters=16*wider_factor, kernel_size=3, pooling_size=1, batch_normalization=False,
                                             dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=32*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=64*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=4, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block9 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=7, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block10 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, batch_normalization=False,
                                              dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
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

class GmlModel3(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        wider_factor = 1
        dropout_rate = 0.5
        self.block0 = LightCurveNetworkBlock(filters=16*wider_factor, kernel_size=3, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=16*wider_factor, kernel_size=3, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=32*wider_factor, kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=32*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=64*wider_factor, kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=64*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block9 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block10 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block11 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block12 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block13 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block14 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block15 = LightCurveNetworkBlock(filters=128*wider_factor, kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block16 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block17 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block18 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=13, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block19 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block20 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block21 = LightCurveNetworkBlock(filters=40*wider_factor, kernel_size=1, pooling_size=1, batch_normalization=False, dropout_rate=0, l2_regularization=l2_regularization, spatial=False)
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
        x = self.block11(x, training=training)
        x = self.block12(x, training=training)
        x = self.block13(x, training=training)
        x = self.block14(x, training=training)
        x = self.block15(x, training=training)
        x = self.block16(x, training=training)
        x = self.block17(x, training=training)
        x = self.block18(x, training=training)
        x = self.block19(x, training=training)
        x = self.block20(x, training=training)
        x = self.block21(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs
    
    
class GmlModel3Narrower(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.01
        wider_factor = 0.5
        dropout_rate = 0.5
        self.block0 = LightCurveNetworkBlock(filters=int(16*wider_factor), kernel_size=3, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=int(16*wider_factor), kernel_size=3, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=int(32*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=int(32*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=int(64*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=int(64*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block9 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block10 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block11 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block12 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block13 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block14 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block15 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block16 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block17 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block18 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=13, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block19 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=1, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block20 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=1, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block21 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=1, pooling_size=1, batch_normalization=False, dropout_rate=0, l2_regularization=l2_regularization, spatial=False)
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
        x = self.block11(x, training=training)
        x = self.block12(x, training=training)
        x = self.block13(x, training=training)
        x = self.block14(x, training=training)
        x = self.block15(x, training=training)
        x = self.block16(x, training=training)
        x = self.block17(x, training=training)
        x = self.block18(x, training=training)
        x = self.block19(x, training=training)
        x = self.block20(x, training=training)
        x = self.block21(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs

class GmlModel3NarrowerNoL2(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_values=1):
        super().__init__()
        l2_regularization = 0.0
        wider_factor = 0.5
        dropout_rate = 0.5
        self.block0 = LightCurveNetworkBlock(filters=int(16*wider_factor), kernel_size=3, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block1 = LightCurveNetworkBlock(filters=int(16*wider_factor), kernel_size=3, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block2 = LightCurveNetworkBlock(filters=int(32*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block3 = LightCurveNetworkBlock(filters=int(32*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block4 = LightCurveNetworkBlock(filters=int(64*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block5 = LightCurveNetworkBlock(filters=int(64*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block6 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block7 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block8 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block9 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block10 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block11 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block12 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block13 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block14 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block15 = LightCurveNetworkBlock(filters=int(128*wider_factor), kernel_size=4, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block16 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=3, pooling_size=2, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
        self.block17 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=3, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block18 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=13, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block19 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=1, pooling_size=1, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block20 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=1, pooling_size=1, batch_normalization=False, dropout_rate=dropout_rate, l2_regularization=l2_regularization, spatial=False)
        self.block21 = LightCurveNetworkBlock(filters=int(40*wider_factor), kernel_size=1, pooling_size=1, batch_normalization=False, dropout_rate=0, l2_regularization=l2_regularization, spatial=False)
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
        x = self.block11(x, training=training)
        x = self.block12(x, training=training)
        x = self.block13(x, training=training)
        x = self.block14(x, training=training)
        x = self.block15(x, training=training)
        x = self.block16(x, training=training)
        x = self.block17(x, training=training)
        x = self.block18(x, training=training)
        x = self.block19(x, training=training)
        x = self.block20(x, training=training)
        x = self.block21(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs