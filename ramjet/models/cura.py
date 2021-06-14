from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Convolution1D, MaxPooling1D, AveragePooling1D

from ramjet.models.components.residual_light_curve_network_block import ResidualLightCurveNetworkBlock, \
    BottleNeckResidualLightCurveNetworkBlock, BottleNeckResidualLightCurveNetworkBlockMainPathRepeat, \
    BottleNeckResidualLightCurveNetworkBlockMainPathRepeatMainPathDropout, \
    BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm, \
    BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNormBnAfterActivations
from ramjet.models.components.selu_residual_light_curve_network_block import SeluResidualLightCurveNetworkBlock


class OldCura(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels))
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


class CuraWider(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8 * 4
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False))
        input_channels = output_channels
        for output_channels in [12 * 4, 16 * 4, 20 * 4, 24 * 4, 28 * 4, 32 * 4, 36 * 4, 40 * 4, 44 * 4, 48 * 4, 64 * 4,
                                128 * 4]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels))
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


class CuraNarrower(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 20, 24, 24, 28, 28, 32, 32, 36, 36]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels))
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


class CuraWithDropout(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
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

class Cura(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class Cursa(Model):
    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(SeluResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(SeluResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(SeluResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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


class CuraFinalAveragePoolNarrower(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNorm(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNormNoDo(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.0))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolWithL2(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5, l2_regularization=0.001))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5, l2_regularization=0.001))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5, l2_regularization=0.001))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerer(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 12, 16, 16, 20, 20, 24, 24, 28, 28, 32, 32]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolSuperNarrowNoDo(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False))
        input_channels = output_channels
        for output_channels in [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolSuperNarrowDo0d25(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False, dropout_rate=0.25))
        input_channels = output_channels
        for output_channels in [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.25))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.25))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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


class CuraFinalAveragePoolMainLineDo0d25(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.25))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 128]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.25))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.25))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerLessDeep(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(1):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNonBottleNeck(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 4
        self.blocks.append(ResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28]:
            self.blocks.append(ResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(ResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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


class CuraFinalAveragePoolNonBottleNeckShallow(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 4
        self.blocks.append(ResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28]:
            self.blocks.append(ResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            # for _ in range(0):
            #     self.blocks.append(ResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNonBottleNeckShallowVeryNarrow(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 2
        self.blocks.append(ResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10]:
            self.blocks.append(ResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            # for _ in range(0):
            #     self.blocks.append(ResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNonBottleNeckShallowVeryNarrowInitialBn(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 2
        self.blocks.append(ResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10]:
            self.blocks.append(ResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            # for _ in range(0):
            #     self.blocks.append(ResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolSuperNarrowNoDoMainPathRepeat(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=False))
        input_channels = output_channels
        for output_channels in [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(output_channels=output_channels))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNormNoDoMainPathRepeat(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.0))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.0))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeat(output_channels=output_channels, dropout_rate=0.0))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNormMainPathRepeatRareMainPathDropout(Model):
    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeatMainPathDropout(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.0))
        input_channels = output_channels
        for index, output_channels in enumerate([12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]):
            if index % 3 == 0:
                dropout_rate = 0.5
            else:
                dropout_rate = 0.0
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeatMainPathDropout(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=dropout_rate))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlockMainPathRepeatMainPathDropout(output_channels=output_channels, dropout_rate=0.0))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolSuperNarrowDo0d25InitialBatchNorm(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True, dropout_rate=0.25))
        input_channels = output_channels
        for output_channels in [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.25))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.25))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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


class CuraFinalAveragePoolNarrowerInitialBatchNormEveryWeightBatchNorm(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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


class CuraFinalAveragePoolNarrowerInitialBatchNormEveryWeightBatchNormNoDo(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNorm(output_channels=output_channels, dropout_rate=0))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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


class CuraFinalAveragePoolNarrowerInitialBatchNormEveryWeightBatchNormAfterActivations(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNormBnAfterActivations(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.5))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNormBnAfterActivations(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlockEveryWeightBatchNormBnAfterActivations(output_channels=output_channels, dropout_rate=0.5))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNormWithL2(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.5, l2_regularization=0.01))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5, l2_regularization=0.01))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5, l2_regularization=0.01))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNormWithL21en3(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.5, l2_regularization=0.001))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5, l2_regularization=0.001))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5, l2_regularization=0.001))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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

class CuraFinalAveragePoolNarrowerInitialBatchNormWithL21en4(Model):
    """
    A general convolutional model for light curve data.
    """

    def __init__(self, number_of_label_types=1, number_of_input_channels: int = 1):
        super().__init__()
        self.blocks = []
        output_channels = 8
        self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
            output_channels=output_channels, input_channels=number_of_input_channels, batch_normalization=True,
            dropout_rate=0.5, l2_regularization=0.0001))
        input_channels = output_channels
        for output_channels in [12, 16, 20, 24, 28, 32, 32, 36, 36, 40, 40, 44]:
            self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(
                output_channels=output_channels, input_channels=input_channels, pooling_size=2, dropout_rate=0.5, l2_regularization=0.0001))
            for _ in range(2):
                self.blocks.append(BottleNeckResidualLightCurveNetworkBlock(output_channels=output_channels, dropout_rate=0.5, l2_regularization=0.0001))
            input_channels = output_channels
        self.final_pooling = AveragePooling1D(pool_size=5)
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