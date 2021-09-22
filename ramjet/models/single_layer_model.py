from tensorflow import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Concatenate


class SingleLayerModel(Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.dense = Dense(1, activation=sigmoid)

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        x = inputs
        x = self.flatten(x)
        x = self.dense(x, training=training)
        outputs = x
        return outputs


class SingleLayerModelWithAuxiliary(Model):
    def __init__(self, number_of_label_values: int = 1):
        super().__init__()
        self.flatten_light_curve = Flatten()
        self.concatenate = Concatenate()
        self.dense = Dense(number_of_label_values, activation=sigmoid)

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the laye1.
        """
        light_curve, auxiliary_information = inputs
        x = self.flatten_light_curve(light_curve)
        x = self.concatenate([x, auxiliary_information])
        x = self.dense(x, training=training)
        outputs = x
        return outputs
