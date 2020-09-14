from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Reshape, Convolution1D

from ramjet.models.components.light_curve_network_block import LightCurveNetworkBlock


class Hades(Model):
    def __init__(self):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=3, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=3)
        self.block2 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=3)
        self.block3 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block7 = LightCurveNetworkBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block8 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, spatial=False)
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