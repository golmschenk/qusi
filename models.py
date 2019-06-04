"""Code for network architectures."""
from tensorflow import sigmoid
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Reshape, LeakyReLU
from tensorflow.python.keras.regularizers import l2


class SanityCheckNetwork(Sequential):
    """A network consisting of a single fully connected layer."""
    def __init__(self):
        super().__init__()
        self.add(Flatten())
        self.add(Dense(1, activation=sigmoid))
        self.add(Reshape([1]))


class SimpleCubeCnn(Sequential):
    """A simple simple 3D CNN for TESS data cubes."""
    def __init__(self):
        super().__init__()
        leaky_relu = LeakyReLU(alpha=0.01)
        l2_regularizer = l2(0.001)
        self.add(Conv3D(16, [3, 3, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(16, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(32, [3, 3, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(32, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(64, [3, 3, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(64, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(128, [4, 4, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(128, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(128, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(128, [1, 1, 4], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(32, [1, 1, 9], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(16, [1, 1, 1], activation=leaky_relu, kernel_regularizer=l2_regularizer))
        self.add(Conv3D(1, [1, 1, 1], activation=sigmoid))
