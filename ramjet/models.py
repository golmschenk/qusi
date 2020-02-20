"""Code for network architectures."""
from tensorflow import sigmoid
from tensorflow.keras import Sequential, Model, backend
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, LeakyReLU, Conv1D, BatchNormalization, \
    LSTM, AveragePooling1D, Layer, Bidirectional, Lambda, Conv2DTranspose
from tensorflow.keras.regularizers import l2


class SanityCheckNetwork(Sequential):
    """A network consisting of a single fully connected layer."""

    def __init__(self):
        super().__init__()
        self.add(Flatten())
        self.add(Dense(1, activation=sigmoid))
        self.add(Reshape([1]))


class SimpleCubeCnn(Sequential):
    """A simple 3D CNN for TESS data cubes."""

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.add(Conv3D(16, [3, 3, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(16, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(32, [3, 3, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(32, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(64, [3, 3, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(64, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(128, [4, 4, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(128, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(128, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(128, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(32, [1, 1, 9], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(16, [1, 1, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(1, [1, 1, 1], activation=sigmoid))
        self.add(Reshape([1]))


class SimpleLightcurveCnn(Model):
    """A simple 1D CNN for lightcurves."""

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.convolution1 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm0 = BatchNormalization(renorm=True)
        self.convolution2 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm1 = BatchNormalization(renorm=True)
        self.convolution3 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm2 = BatchNormalization(renorm=True)
        self.convolution4 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm3 = BatchNormalization(renorm=True)
        self.convolution5 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.convolution6 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm5 = BatchNormalization(renorm=True)
        self.convolution7 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm6 = BatchNormalization(renorm=True)
        self.convolution8 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm7 = BatchNormalization(renorm=True)
        self.convolution9 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm8 = BatchNormalization(renorm=True)
        self.convolution10 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                    kernel_regularizer=l2_regularizer)
        self.batch_norm9 = BatchNormalization(renorm=True)
        self.convolution11 = Conv1D(10, kernel_size=12, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolution12 = Conv1D(1, [1], activation=sigmoid)
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
        x = self.convolution0(x)
        x = self.convolution1(x)
        x = self.batch_norm0(x, training=training)
        x = self.convolution2(x)
        x = self.batch_norm1(x, training=training)
        x = self.convolution3(x)
        x = self.batch_norm2(x, training=training)
        x = self.convolution4(x)
        x = self.batch_norm3(x, training=training)
        x = self.convolution5(x)
        x = self.batch_norm4(x, training=training)
        x = self.convolution6(x)
        x = self.batch_norm5(x, training=training)
        x = self.convolution7(x)
        x = self.batch_norm6(x, training=training)
        x = self.convolution8(x)
        x = self.batch_norm7(x, training=training)
        x = self.convolution9(x)
        x = self.batch_norm8(x, training=training)
        x = self.convolution10(x)
        x = self.batch_norm9(x, training=training)
        x = self.convolution11(x)
        x = self.convolution12(x)
        x = self.reshape(x)
        return x


class SimpleLightcurveLstm(Model):
    """A simple LSTM model for lightcurves."""

    def __init__(self):
        super().__init__()
        self.lstm0 = LSTM(10, return_sequences=True)
        self.lstm1 = LSTM(20, return_sequences=True)
        self.lstm2 = LSTM(30, return_sequences=True)
        self.convolution0 = Conv1D(1, kernel_size=1, activation=sigmoid)

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.lstm0(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.convolution0(x)
        return x


class SimpleLightcurveCnnPerTimeStepLabel(Model):
    """A simple 1D CNN for lightcurves."""

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(10, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.convolution1 = Conv1D(10, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm0 = BatchNormalization(renorm=True)
        self.max_pool0 = AveragePooling1D(pool_size=5, strides=1, padding='same')
        self.convolution2 = Conv1D(10, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm1 = BatchNormalization(renorm=True)
        self.max_pool1 = AveragePooling1D(pool_size=10, strides=1, padding='same')
        self.convolution3 = Conv1D(20, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm2 = BatchNormalization(renorm=True)
        self.max_pool2 = AveragePooling1D(pool_size=10, strides=1, padding='same')
        self.convolution4 = Conv1D(20, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm3 = BatchNormalization(renorm=True)
        self.max_pool3 = AveragePooling1D(pool_size=25, strides=1, padding='same')
        self.convolution5 = Conv1D(20, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.max_pool4 = AveragePooling1D(pool_size=25, strides=1, padding='same')
        self.convolution6 = Conv1D(30, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm5 = BatchNormalization(renorm=True)
        self.max_pool5 = AveragePooling1D(pool_size=50, strides=1, padding='same')
        self.convolution7 = Conv1D(30, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm6 = BatchNormalization(renorm=True)
        self.max_pool6 = AveragePooling1D(pool_size=50, strides=1, padding='same')
        self.convolution8 = Conv1D(30, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.batch_norm7 = BatchNormalization(renorm=True)
        self.max_pool7 = AveragePooling1D(pool_size=100, strides=1, padding='same')
        self.convolution9 = Conv1D(10, kernel_size=5, activation=leaky_relu, kernel_regularizer=l2_regularizer,
                                   padding='same')
        self.max_pool8 = AveragePooling1D(pool_size=100, strides=1, padding='same')
        self.convolution10 = Conv1D(1, kernel_size=5, activation=sigmoid, padding='same')
        self.reshape = Reshape([-1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.convolution0(x)
        x = self.convolution1(x)
        x = self.batch_norm0(x, training=training)
        x = self.max_pool0(x)
        x = self.convolution2(x)
        x = self.batch_norm1(x, training=training)
        x = self.max_pool1(x)
        x = self.convolution3(x)
        x = self.batch_norm2(x, training=training)
        x = self.max_pool2(x)
        x = self.convolution4(x)
        x = self.batch_norm3(x, training=training)
        x = self.max_pool3(x)
        x = self.convolution5(x)
        x = self.batch_norm4(x, training=training)
        x = self.max_pool4(x)
        x = self.convolution6(x)
        x = self.batch_norm5(x, training=training)
        x = self.max_pool5(x)
        x = self.convolution7(x)
        x = self.batch_norm6(x, training=training)
        x = self.max_pool6(x)
        x = self.convolution8(x)
        x = self.batch_norm7(x, training=training)
        x = self.max_pool7(x)
        x = self.convolution9(x)
        x = self.max_pool8(x)
        x = self.convolution10(x)
        x = self.reshape(x)
        return x


class Conv1DTranspose(Layer):
    """
    A 1D transposed convolutional layer.
    """

    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        super().__init__()
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        self._model = Sequential()

    def build(self, input_shape):
        """
        Builds the layer.

        :param input_shape: The input tensor shape.
        """
        self._model.add(Lambda(lambda x: backend.expand_dims(x, axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:, 0]))
        super().build(input_shape)

    def call(self, x, training=False, mask=None):
        """
        The forward pass of the layer.

        :param x: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        return self._model(x)

    def compute_output_shape(self, input_shape):
        """
        The output shape of the layer.

        :param input_shape:
        :return:
        """
        return self._model.compute_output_shape(input_shape)


class ConvolutionalLstm(Model):
    """
    A convolutional LSTM network.
    """

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer, padding='same')
        self.convolution1 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer, padding='same')
        self.batch_norm_c1 = BatchNormalization(renorm=True)
        self.convolution2 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer, padding='same')
        self.batch_norm_c2 = BatchNormalization(renorm=True)
        self.convolution3 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer, padding='same')
        self.batch_norm_c3 = BatchNormalization(renorm=True)
        self.convolution4 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer, padding='same')
        self.lstm0 = Bidirectional(LSTM(64, return_sequences=True))
        self.lstm1 = Bidirectional(LSTM(64, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(64, return_sequences=True))
        self.transposed_convolution0 = Conv1DTranspose(64, kernel_size=4, strides=2, activation=leaky_relu,
                                                       kernel_regularizer=l2_regularizer, padding='same')
        self.batch_norm_t0 = BatchNormalization(renorm=True)
        self.transposed_convolution1 = Conv1DTranspose(64, kernel_size=4, strides=2, activation=leaky_relu,
                                                       kernel_regularizer=l2_regularizer, padding='same')
        self.batch_norm_t1 = BatchNormalization(renorm=True)
        self.transposed_convolution2 = Conv1DTranspose(32, kernel_size=4, strides=2, activation=leaky_relu,
                                                       kernel_regularizer=l2_regularizer, padding='same')
        self.batch_norm_t2 = BatchNormalization(renorm=True)
        self.transposed_convolution3 = Conv1DTranspose(16, kernel_size=4, strides=2, activation=leaky_relu,
                                                       kernel_regularizer=l2_regularizer, padding='same')
        self.transposed_convolution4 = Conv1DTranspose(8, kernel_size=4, strides=2, activation=leaky_relu,
                                                       kernel_regularizer=l2_regularizer, padding='same')
        self.prediction_layer = Conv1D(1, kernel_size=1, activation=sigmoid)
        self.reshape = Reshape([-1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.convolution0(x)
        x = self.convolution1(x)
        x = self.batch_norm_c1(x)
        x = self.convolution2(x)
        x = self.batch_norm_c2(x)
        x = self.convolution3(x)
        x = self.batch_norm_c3(x)
        x = self.convolution4(x)
        x = self.lstm0(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.transposed_convolution0(x)
        x = self.batch_norm_t0(x)
        x = self.transposed_convolution1(x)
        x = self.batch_norm_t1(x)
        x = self.transposed_convolution2(x)
        x = self.batch_norm_t2(x)
        x = self.transposed_convolution3(x)
        x = self.transposed_convolution4(x)
        x = self.prediction_layer(x)
        x = self.reshape(x)
        return x
