"""Code for network architectures."""
from tensorflow import sigmoid
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv3D, LeakyReLU, MaxPool3D, Dropout, BatchNormalization, Flatten, Dense, \
    Reshape


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
        dropout_rate = 0.5
        self.add(Conv3D(8, [3, 3, 1]))
        self.add(LeakyReLU(alpha=0.01))
        self.add(Conv3D(8, [1, 1, 4]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(Dropout(dropout_rate))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(16, [3, 3, 1]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(BatchNormalization())
        # self.add(Dropout(dropout_rate))
        self.add(Conv3D(16, [1, 1, 4]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(BatchNormalization())
        # self.add(Dropout(dropout_rate))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(32, [3, 3, 1]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(BatchNormalization())
        self.add(Dropout(dropout_rate))
        self.add(Conv3D(32, [1, 1, 4]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(BatchNormalization())
        self.add(Dropout(dropout_rate))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(64, [4, 4, 1]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(BatchNormalization())
        self.add(Dropout(dropout_rate))
        self.add(Conv3D(64, [1, 1, 4]))
        self.add(LeakyReLU(alpha=0.01))
        self.add(Dropout(dropout_rate))
        # self.add(BatchNormalization())
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(64, [1, 1, 4]))
        self.add(LeakyReLU(alpha=0.01))
        # self.add(BatchNormalization())
        self.add(Dropout(dropout_rate))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(64, [1, 1, 4]))
        self.add(LeakyReLU(alpha=0.01))
        self.add(Dropout(dropout_rate))
        self.add(MaxPool3D([1, 1, 2]))
        self.add(Conv3D(16, [1, 1, 9]))
        self.add(LeakyReLU(alpha=0.01))
        self.add(Dropout(dropout_rate))
        self.add(Conv3D(1, [1, 1, 1], activation=sigmoid))
