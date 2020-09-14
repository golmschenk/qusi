from tensorflow.keras.layers import LeakyReLU, Conv1D, SpatialDropout1D, Dropout, MaxPooling1D, BatchNormalization,\
    Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Reshape


class LightCurveNetworkBlock(Layer):
    """A block containing a convolution and all the fixings that go with it."""
    def __init__(self, filters: int, kernel_size: int, pooling_size: int, dropout_rate: float = 0.1,
                 batch_normalization: bool = True, spatial: bool = True):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        self.convolution = Conv1D(filters, kernel_size=kernel_size, activation=leaky_relu)
        if dropout_rate > 0:
            if spatial:
                self.dropout = SpatialDropout1D(dropout_rate)
            else:
                self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None
        if pooling_size > 1:
            self.max_pooling = MaxPooling1D(pool_size=pooling_size)
        else:
            self.max_pooling = None
        if batch_normalization:
            self.batch_normalization = BatchNormalization(scale=False)
            if not spatial:
                self.batch_normalization_input_reshape = Reshape([-1])
                self.batch_normalization_output_reshape = Reshape([-1, filters])
            else:
                self.batch_normalization_input_reshape = None
                self.batch_normalization_output_reshape = None
        else:
            self.batch_normalization = None

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
        if self.max_pooling is not None:
            x = self.max_pooling(x, training=training)
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x)
            x = self.batch_normalization(x, training=training)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x)
        return x
