import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops


class PerTimeStepBinaryCrossEntropy(LossFunctionWrapper):
    """
    Computes the cross-entropy loss between true labels and predicted labels for a time series which each time step has
    a binary label.
    """
    def __init__(self, positive_weight: float = 1, *args, **kwargs):
        """
        :param positive_weight: The weight to give to positive labels in calculating the loss (relative to negative).
        """
        super().__init__(self.per_time_step_binary_cross_entropy, positive_weight=positive_weight, *args, **kwargs)
        self.positive_weight = positive_weight

    @staticmethod
    def per_time_step_binary_cross_entropy(y_true: tf.Tensor, y_pred: tf.Tensor, positive_weight: float = 1
                                           ) -> tf.Tensor:
        """
        Calculates the cross-entropy loss between true labels and predicted labels for a time series which each time
        step has a binary label.

        :param y_true: The true label.
        :param y_pred: The predicted label.
        :param positive_weight: The weight to give to positive labels in calculating the loss (relative to negative).
        :return: The resulting cross entropy loss.
        """
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        binary_cross_entropy = K.binary_crossentropy(y_true, y_pred)
        weights = tf.where(tf.cast(y_true, dtype=tf.bool), tf.cast(positive_weight, dtype=tf.float32),
                           tf.cast(1, dtype=tf.float32))
        return binary_cross_entropy * weights
