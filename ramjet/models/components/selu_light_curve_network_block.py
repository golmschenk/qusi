from tensorflow.keras.layers import Conv1D, AlphaDropout, MaxPooling1D, Layer, AveragePooling1D
from tensorflow.keras.regularizers import l2


class SeluLightCurveNetworkBlock(Layer):
    """A block containing a SELU convolution and all the fixings that go with it."""
    def __init__(self, filters: int, kernel_size: int, pooling_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.convolution = Conv1D(filters, kernel_size=kernel_size,
                                  kernel_initializer='lecun_normal', activation='selu')
        if dropout_rate > 0:
            self.dropout = AlphaDropout(dropout_rate, [100, 1, filters])
        else:
            self.dropout = None
        if pooling_size > 1:
            self.max_pooling = MaxPooling1D(pool_size=pooling_size)
        else:
            self.max_pooling = None

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.convolution(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        if self.pooling is not None:
            x = self.pooling(x, training=training)
        return x
