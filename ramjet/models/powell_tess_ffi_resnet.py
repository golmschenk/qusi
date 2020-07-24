"""
Code for Brian Powell's ResNet.
"""
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, LeakyReLU, Reshape, Dense, Dropout, Flatten, \
    Multiply, Add

lcsize = 1400


def resblockconv(channels, inputlayer):
    x = Conv1D(channels, kernel_size=4, strides=2, padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return x


def resblockconvsc(channels, inputlayer):
    x = Conv1D(channels, kernel_size=4, strides=2, padding='same')(inputlayer)
    x = BatchNormalization()(x)
    return x


def resblockid(channels, inputlayer):
    x = Conv1D(channels, kernel_size=8, strides=1, padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return x


def fullresblock(channels, inputlayer):
    xr = resblockconv(channels, inputlayer)
    xrsc = resblockconvsc(channels, inputlayer)
    xadd = Add()([xr, xrsc])
    xact = LeakyReLU(alpha=.1)(xadd)
    xid = resblockid(channels, xact)
    xadd = Add()([xact, xid])
    xact = LeakyReLU(alpha=.1)(xadd)
    return xact

def network():
    input_layer = Input(shape=(lcsize, 1))
    x = Reshape((lcsize,))(input_layer)
    attn = Dense(lcsize, activation='softmax')(x)
    mult = Multiply()([attn, x])
    x = Add()([mult, x])
    x = Reshape((lcsize, 1))(x)
    x = fullresblock(16, x)
    x = Dropout(.2)(x)
    x = fullresblock(32, x)
    x = Dropout(.2)(x)
    x = fullresblock(48, x)
    x = Dropout(.2)(x)
    x = fullresblock(54, x)
    x = Dropout(.2)(x)
    x = fullresblock(70, x)
    x = Dropout(.2)(x)
    x = fullresblock(86, x)
    x = Dropout(.2)(x)
    x = fullresblock(102, x)
    x = Dropout(.2)(x)
    x = fullresblock(118, x)
    x = Dropout(.2)(x)
    x = fullresblock(134, x)
    x = Flatten()(x)
    x = Dropout(.2)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Dropout(.2)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
