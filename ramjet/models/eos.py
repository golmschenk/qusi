from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Convolution1D, Reshape

from ramjet.models.components.selu_light_curve_network_block import SeluLightCurveNetworkBlock


class Eos(Model):
    def __init__(self):
        super().__init__()
        self.block0 = SeluLightCurveNetworkBlock(filters=5, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block1 = SeluLightCurveNetworkBlock(filters=10, kernel_size=3, pooling_size=2)
        self.block2 = SeluLightCurveNetworkBlock(filters=10, kernel_size=3, pooling_size=2)
        self.block3 = SeluLightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1)
        self.block4 = SeluLightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2)
        self.block5 = SeluLightCurveNetworkBlock(filters=30, kernel_size=3, pooling_size=1)
        self.block6 = SeluLightCurveNetworkBlock(filters=30, kernel_size=3, pooling_size=2)
        self.block7 = SeluLightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=1)
        self.block8 = SeluLightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=2)
        self.block9 = SeluLightCurveNetworkBlock(filters=50, kernel_size=3, pooling_size=1)
        self.block10 = SeluLightCurveNetworkBlock(filters=50, kernel_size=3, pooling_size=2)
        self.block11 = SeluLightCurveNetworkBlock(filters=60, kernel_size=3, pooling_size=1)
        self.block12 = SeluLightCurveNetworkBlock(filters=60, kernel_size=3, pooling_size=2)
        self.block13 = SeluLightCurveNetworkBlock(filters=70, kernel_size=3, pooling_size=1)
        self.block14 = SeluLightCurveNetworkBlock(filters=70, kernel_size=3, pooling_size=2)
        self.block15 = SeluLightCurveNetworkBlock(filters=80, kernel_size=3, pooling_size=1)
        self.block16 = SeluLightCurveNetworkBlock(filters=80, kernel_size=3, pooling_size=2)
        self.block17 = SeluLightCurveNetworkBlock(filters=90, kernel_size=3, pooling_size=1)
        self.block18 = SeluLightCurveNetworkBlock(filters=90, kernel_size=3, pooling_size=2)
        self.block19 = SeluLightCurveNetworkBlock(filters=100, kernel_size=3, pooling_size=1)
        self.block20 = SeluLightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1)
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
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs
