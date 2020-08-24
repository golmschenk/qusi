from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Convolution1D, Reshape

from ramjet.basic_models import LightCurveNetworkBlock


class Nyx(Model):
    def __init__(self):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=5, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=10, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block2 = LightCurveNetworkBlock(filters=10, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block3 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=1, dropout_rate=0)
        self.block4 = LightCurveNetworkBlock(filters=20, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block5 = LightCurveNetworkBlock(filters=30, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block6 = LightCurveNetworkBlock(filters=30, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block7 = LightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block8 = LightCurveNetworkBlock(filters=40, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block9 = LightCurveNetworkBlock(filters=50, kernel_size=3, pooling_size=1, dropout_rate=0)
        self.block10 = LightCurveNetworkBlock(filters=50, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block11 = LightCurveNetworkBlock(filters=60, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block12 = LightCurveNetworkBlock(filters=60, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block13 = LightCurveNetworkBlock(filters=70, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block14 = LightCurveNetworkBlock(filters=70, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block15 = LightCurveNetworkBlock(filters=80, kernel_size=3, pooling_size=1, dropout_rate=0)
        self.block16 = LightCurveNetworkBlock(filters=80, kernel_size=3, pooling_size=2, batch_normalization=False)
        self.block17 = LightCurveNetworkBlock(filters=90, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block18 = LightCurveNetworkBlock(filters=90, kernel_size=3, pooling_size=2, dropout_rate=0)
        self.block19 = LightCurveNetworkBlock(filters=100, kernel_size=3, pooling_size=1, batch_normalization=False)
        self.block20 = LightCurveNetworkBlock(filters=20, kernel_size=1, pooling_size=1, batch_normalization=False)
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