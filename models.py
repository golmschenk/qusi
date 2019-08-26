"""Code for network architectures."""
from tensorflow import sigmoid
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, LeakyReLU, Conv1D, \
    BatchNormalization
from tensorflow.python.keras.regularizers import l2


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
        leaky_relu.__name__ = 'LeakyReLU'  # Fix bug in Keras model saving requiring layer name for activation.
        l2_regularizer = l2(0.001)
        self.convolution0 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolution1 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm0 = BatchNormalization(renorm=True)
        self.convolution2 = Conv1D(8, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm1 = BatchNormalization(renorm=True)
        self.convolution3 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm2 = BatchNormalization(renorm=True)
        self.convolution4 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm3 = BatchNormalization(renorm=True)
        self.convolution5 = Conv1D(16, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm4 = BatchNormalization(renorm=True)
        self.convolution6 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm5 = BatchNormalization(renorm=True)
        self.convolution7 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm6 = BatchNormalization(renorm=True)
        self.convolution8 = Conv1D(32, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm7 = BatchNormalization(renorm=True)
        self.convolution9 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm8 = BatchNormalization(renorm=True)
        self.convolution10 = Conv1D(64, kernel_size=4, strides=2, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.batch_norm9 = BatchNormalization(renorm=True)
        self.convolution11 = Conv1D(10, kernel_size=12, activation=leaky_relu, kernel_regularizer=l2_regularizer)
        self.convolution12 = Conv1D(1, [1], activation=sigmoid)
        self.reshape = Reshape([1])

    def call(self, inputs, training=False, mask=None):
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
