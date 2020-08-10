from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, Convolution1D, Reshape
from tensorflow.keras.regularizers import l2

from ramjet.basic_models import ConvolutionPoolingBatchNormalizationBlock


class Hades(Model):
    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.block0 = ConvolutionPoolingBatchNormalizationBlock(filters=8, kernel_size=3, pooling_size=3)
        self.block1 = ConvolutionPoolingBatchNormalizationBlock(filters=8, kernel_size=3, pooling_size=3)
        self.block2 = ConvolutionPoolingBatchNormalizationBlock(filters=16, kernel_size=3, pooling_size=3)
        self.block3 = ConvolutionPoolingBatchNormalizationBlock(filters=32, kernel_size=3, pooling_size=2)
        self.block4 = ConvolutionPoolingBatchNormalizationBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = ConvolutionPoolingBatchNormalizationBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block6 = ConvolutionPoolingBatchNormalizationBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block7 = ConvolutionPoolingBatchNormalizationBlock(filters=128, kernel_size=3, pooling_size=2)
        self.block8 = ConvolutionPoolingBatchNormalizationBlock(filters=128, kernel_size=3, pooling_size=2)
        self.dense0 = Convolution1D(20, kernel_size=9, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.dense1 = Convolution1D(50, kernel_size=1, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.dense2 = Convolution1D(1, kernel_size=1, activation=sigmoid)
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
        x = self.dense0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs