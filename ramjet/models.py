"""Code for network architectures."""
from tensorflow import sigmoid
from tensorflow.keras import Sequential, Model, backend
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, LeakyReLU, Conv1D, BatchNormalization, \
    LSTM, AveragePooling1D, Layer, Bidirectional, Lambda, Conv2DTranspose, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate


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
        self.batch_norm0 = BatchNormalization()
        self.convolution2 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm1 = BatchNormalization()
        self.convolution3 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm2 = BatchNormalization()
        self.convolution4 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm3 = BatchNormalization()
        self.convolution5 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm4 = BatchNormalization()
        self.convolution6 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm5 = BatchNormalization()
        self.convolution7 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm6 = BatchNormalization()
        self.convolution8 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm7 = BatchNormalization()
        self.convolution9 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm8 = BatchNormalization()
        self.convolution10 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                    kernel_regularizer=l2_regularizer)
        self.batch_norm9 = BatchNormalization()
        self.convolution11 = Conv1D(10, kernel_size=7, activation=leaky_relu, kernel_regularizer=l2_regularizer)
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
        x = self.convolution0(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.batch_norm0(x, training=training)
        x = self.convolution2(x, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.convolution3(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.convolution4(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.convolution5(x, training=training)
        x = self.batch_norm4(x, training=training)
        x = self.convolution6(x, training=training)
        x = self.batch_norm5(x, training=training)
        x = self.convolution7(x, training=training)
        x = self.batch_norm6(x, training=training)
        x = self.convolution8(x, training=training)
        x = self.batch_norm7(x, training=training)
        x = self.convolution9(x, training=training)
        x = self.batch_norm8(x, training=training)
        x = self.convolution10(x, training=training)
        x = self.batch_norm9(x, training=training)
        x = self.convolution11(x, training=training)
        x = self.convolution12(x, training=training)
        x = self.reshape(x, training=training)
        return x


class SimpleFfiLightcurveCnn(Model):
    """A simple 1D CNN for FFI lightcurves."""

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
        self.convolution5 = Conv1D(16, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.convolution6 = Conv1D(32, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm5 = BatchNormalization(renorm=True)
        self.convolution7 = Conv1D(32, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm6 = BatchNormalization(renorm=True)
        self.convolution8 = Conv1D(32, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm7 = BatchNormalization(renorm=True)
        self.convolution9 = Conv1D(64, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm8 = BatchNormalization(renorm=True)
        self.convolution10 = Conv1D(64, kernel_size=4, activation=leaky_relu,
                                    kernel_regularizer=l2_regularizer)
        self.batch_norm9 = BatchNormalization(renorm=True)
        self.convolution11 = Conv1D(10, kernel_size=20, activation=leaky_relu, kernel_regularizer=l2_regularizer)
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
        x = self.convolution0(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.batch_norm0(x, training=training)
        x = self.convolution2(x, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.convolution3(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.convolution4(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.convolution5(x, training=training)
        x = self.batch_norm4(x, training=training)
        x = self.convolution6(x, training=training)
        x = self.batch_norm5(x, training=training)
        x = self.convolution7(x, training=training)
        x = self.batch_norm6(x, training=training)
        x = self.convolution8(x, training=training)
        x = self.batch_norm7(x, training=training)
        x = self.convolution9(x, training=training)
        x = self.batch_norm8(x, training=training)
        x = self.convolution10(x, training=training)
        x = self.batch_norm9(x, training=training)
        x = self.convolution11(x, training=training)
        x = self.convolution12(x, training=training)
        x = self.reshape(x, training=training)
        return x


class SmallFfiLightcurveCnn(Model):
    """A simple 1D CNN for FFI lightcurves."""

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.convolution1 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm0 = BatchNormalization(renorm=True)
        self.convolution2 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm1 = BatchNormalization(renorm=True)
        self.convolution3 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm2 = BatchNormalization(renorm=True)
        self.convolution4 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm3 = BatchNormalization(renorm=True)
        self.convolution5 = Conv1D(128, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.convolution6 = Conv1D(10, kernel_size=18, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolution7 = Conv1D(1, [1], activation=sigmoid)
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
        x = self.convolution0(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.batch_norm0(x, training=training)
        x = self.convolution2(x, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.convolution3(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.convolution4(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.convolution5(x, training=training)
        x = self.batch_norm4(x, training=training)
        x = self.convolution6(x, training=training)
        x = self.convolution7(x, training=training)
        x = self.reshape(x, training=training)
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
        x = self.lstm0(x, training=training)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.convolution0(x, training=training)
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
        x = self.convolution0(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.batch_norm0(x, training=training)
        x = self.max_pool0(x, training=training)
        x = self.convolution2(x, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.max_pool1(x, training=training)
        x = self.convolution3(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.max_pool2(x, training=training)
        x = self.convolution4(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.max_pool3(x, training=training)
        x = self.convolution5(x, training=training)
        x = self.batch_norm4(x, training=training)
        x = self.max_pool4(x, training=training)
        x = self.convolution6(x, training=training)
        x = self.batch_norm5(x, training=training)
        x = self.max_pool5(x, training=training)
        x = self.convolution7(x, training=training)
        x = self.batch_norm6(x, training=training)
        x = self.max_pool6(x, training=training)
        x = self.convolution8(x, training=training)
        x = self.batch_norm7(x, training=training)
        x = self.max_pool7(x, training=training)
        x = self.convolution9(x, training=training)
        x = self.max_pool8(x, training=training)
        x = self.convolution10(x, training=training)
        x = self.reshape(x, training=training)
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
        return self._model(x, training=training)

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
        x = self.convolution0(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.batch_norm_c1(x, training=training)
        x = self.convolution2(x, training=training)
        x = self.batch_norm_c2(x, training=training)
        x = self.convolution3(x, training=training)
        x = self.batch_norm_c3(x, training=training)
        x = self.convolution4(x, training=training)
        x = self.lstm0(x, training=training)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.transposed_convolution0(x, training=training)
        x = self.batch_norm_t0(x, training=training)
        x = self.transposed_convolution1(x, training=training)
        x = self.batch_norm_t1(x, training=training)
        x = self.transposed_convolution2(x, training=training)
        x = self.batch_norm_t2(x, training=training)
        x = self.transposed_convolution3(x, training=training)
        x = self.transposed_convolution4(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = self.reshape(x, training=training)
        return x


class ConvolutionalLstmMeanFinal(Model):

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(4, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.convolution1 = Conv1D(4, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm0 = BatchNormalization(renorm=True)
        self.convolution2 = Conv1D(4, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm1 = BatchNormalization(renorm=True)
        self.convolution3 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm2 = BatchNormalization(renorm=True)
        self.convolution4 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm3 = BatchNormalization(renorm=True)
        self.convolution5 = Conv1D(8, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.convolution6 = Conv1D(16, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm5 = BatchNormalization(renorm=True)
        self.convolution7 = Conv1D(16, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.batch_norm6 = BatchNormalization(renorm=True)
        self.convolution8 = Conv1D(16, kernel_size=4, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm0 = Bidirectional(LSTM(10, return_sequences=True))
        self.lstm1 = Bidirectional(LSTM(10, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(10, return_sequences=True))
        self.prediction_layer = Conv1D(1, kernel_size=1, activation=sigmoid)
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
        x = self.convolution0(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.batch_norm0(x, training=training)
        x = self.convolution2(x, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.convolution3(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.convolution4(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.convolution5(x, training=training)
        x = self.batch_norm4(x, training=training)
        x = self.convolution6(x, training=training)
        x = self.batch_norm5(x, training=training)
        x = self.convolution7(x, training=training)
        x = self.batch_norm6(x, training=training)
        x = self.convolution8(x, training=training)
        x = self.lstm0(x, training=training)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = backend.mean(x, axis=[1, 2])
        x = self.reshape(x)
        return x


class SimpleLightcurveCnnWithLstmLayers(Model):
    """A simple 1D CNN for lightcurves."""

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.convolution1 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm1 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm1 = BatchNormalization(renorm=True)
        self.concatenate1 = Concatenate()
        self.convolution2 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm2 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm2 = BatchNormalization(renorm=True)
        self.concatenate2 = Concatenate()
        self.convolution3 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm3 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm3 = BatchNormalization(renorm=True)
        self.concatenate3 = Concatenate()
        self.convolution4 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm4 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.concatenate4 = Concatenate()
        self.convolution5 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm5 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm5 = BatchNormalization(renorm=True)
        self.concatenate5 = Concatenate()
        self.convolution6 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm6 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm6 = BatchNormalization(renorm=True)
        self.concatenate6 = Concatenate()
        self.convolution7 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm7 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm7 = BatchNormalization(renorm=True)
        self.concatenate7 = Concatenate()
        self.convolution8 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm8 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm8 = BatchNormalization(renorm=True)
        self.concatenate8 = Concatenate()
        self.convolution9 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                   kernel_regularizer=l2_regularizer)
        self.lstm9 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm9 = BatchNormalization(renorm=True)
        self.concatenate9 = Concatenate()
        self.convolution10 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu,
                                    kernel_regularizer=l2_regularizer)
        self.lstm10 = Bidirectional(LSTM(10, return_sequences=True))
        self.batch_norm10 = BatchNormalization(renorm=True)
        self.concatenate10 = Concatenate()
        self.convolution11 = Conv1D(10, kernel_size=7, activation=leaky_relu, kernel_regularizer=l2_regularizer)
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
        x0 = self.convolution0(x, training=training)
        x1c = self.convolution1(x0, training=training)
        x1b = self.batch_norm1(x1c, training=training)
        x1l = self.lstm1(x1c, training=training)
        x1 = self.concatenate1([x1b, x1l], training=training)
        x2c = self.convolution2(x1, training=training)
        x2b = self.batch_norm2(x2c, training=training)
        x2l = self.lstm2(x2c, training=training)
        x2 = self.concatenate2([x2b, x2l], training=training)
        x3c = self.convolution3(x2, training=training)
        x3b = self.batch_norm3(x3c, training=training)
        x3l = self.lstm3(x3c, training=training)
        x3 = self.concatenate3([x3b, x3l], training=training)
        x4c = self.convolution4(x3, training=training)
        x4b = self.batch_norm4(x4c, training=training)
        x4l = self.lstm4(x4c, training=training)
        x4 = self.concatenate4([x4b, x4l], training=training)
        x5c = self.convolution5(x4, training=training)
        x5b = self.batch_norm5(x5c, training=training)
        x5l = self.lstm5(x5c, training=training)
        x5 = self.concatenate5([x5b, x5l], training=training)
        x6c = self.convolution6(x5, training=training)
        x6b = self.batch_norm6(x6c, training=training)
        x6l = self.lstm6(x6c, training=training)
        x6 = self.concatenate6([x6b, x6l], training=training)
        x7c = self.convolution7(x6, training=training)
        x7b = self.batch_norm7(x7c, training=training)
        x7l = self.lstm7(x7c, training=training)
        x7 = self.concatenate7([x7b, x7l], training=training)
        x8c = self.convolution8(x7, training=training)
        x8b = self.batch_norm8(x8c, training=training)
        x8l = self.lstm8(x8c, training=training)
        x8 = self.concatenate8([x8b, x8l], training=training)
        x9c = self.convolution9(x8, training=training)
        x9b = self.batch_norm9(x9c, training=training)
        x9l = self.lstm9(x9c, training=training)
        x9 = self.concatenate9([x9b, x9l], training=training)
        x10c = self.convolution10(x9, training=training)
        x10b = self.batch_norm10(x10c, training=training)
        x10l = self.lstm10(x10c, training=training)
        x10 = self.concatenate10([x10b, x10l], training=training)
        x11 = self.convolution11(x10, training=training)
        x12 = self.convolution12(x11, training=training)
        output = self.reshape(x12, training=training)
        return output


class ResnetBlock1D(Layer):
    def __init__(self, layers: int, channels: int, kernel_size: int, strides=2):
        super().__init__()
        self.layers: int = layers
        self.channels: int = channels
        self.kernel_size: int = kernel_size
        self.strides: int = strides
        self.inner_sequential = Sequential()
        self.skip_sequential = Sequential()
        self.activation = LeakyReLU(alpha=0.01)
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        for _ in range(self.layers - 1):
            self.inner_sequential.add(Conv1D(self.channels, kernel_size=self.kernel_size, strides=self.strides,
                                             activation=leaky_relu, kernel_regularizer=l2_regularizer, padding='SAME'))
            self.inner_sequential.add(BatchNormalization())
        self.inner_sequential.add(Conv1D(self.channels, kernel_size=self.kernel_size, strides=self.strides,
                                         kernel_regularizer=l2_regularizer, padding='SAME'))
        self.inner_sequential.add(BatchNormalization())
        self.skip_sequential.add(Conv1D(self.channels, kernel_size=1, strides=self.strides ** self.layers,
                                        kernel_regularizer=l2_regularizer, padding='SAME'))
        self.skip_sequential.add(BatchNormalization())

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        inner_output = self.inner_sequential(inputs, training=training)
        skip_output = self.skip_sequential(inputs, training=training)
        output = self.activation(add([inner_output, skip_output]), training=training)
        return output


class SimpleLightcurveCnnWithSkipConnections(Model):
    """A simple 1D CNN for lightcurves."""

    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.resnet_block0 = ResnetBlock1D(layers=3, channels=8, kernel_size=4, strides=2)
        self.resnet_block1 = ResnetBlock1D(layers=3, channels=16, kernel_size=4, strides=2)
        self.resnet_block2 = ResnetBlock1D(layers=3, channels=32, kernel_size=4, strides=2)
        self.resnet_block3 = ResnetBlock1D(layers=2, channels=64, kernel_size=4, strides=2)
        self.end_convolution0 = Conv1D(10, kernel_size=10, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.end_convolution1 = Conv1D(1, [1], activation=sigmoid)
        self.reshape = Reshape([1])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        resnet_block0_outputs = self.resnet_block0(inputs, training=training)
        resnet_block1_outputs = self.resnet_block1(resnet_block0_outputs, training=training)
        resnet_block2_outputs = self.resnet_block2(resnet_block1_outputs, training=training)
        resnet_block3_outputs = self.resnet_block3(resnet_block2_outputs, training=training)
        end_convolution0_outputs = self.end_convolution0(resnet_block3_outputs, training=training)
        end_convolution1_outputs = self.end_convolution1(end_convolution0_outputs, training=training)
        outputs = self.reshape(end_convolution1_outputs, training=training)
        return outputs
