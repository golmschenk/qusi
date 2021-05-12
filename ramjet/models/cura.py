from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, MaxPooling1D

from ramjet.models.components.residual_light_curve_network_block import ResidualLightCurveNetworkBlock


class Cura(Model):
    """
    A general convolutional model for light curve data.
    """
    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(ResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(ResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2))
            for _ in range(2):
                self.blocks.append(ResidualLightCurveNetworkBlock(output_channels=output_channels))
            input_channels = output_channels
        self.final_pooling = MaxPooling1D(pool_size=5)
        self.prediction_layer = Convolution1D(number_of_label_types, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([number_of_label_types])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
        x = self.final_pooling(x, training=training)
        x = self.prediction_layer(x, training=training)
        outputs = self.reshape(x, training=training)
        return outputs
