"""
Code for a class to calculate receiver operating characteristic (ROC) curves.
"""
import numpy as np


class RocCalculator:
    """
    A class to calculate receiver operating characteristic (ROC) curves.
    """

    def threshold_predictions(self, probability_predictions: np.ndarray, thresholds: np.ndarray):
        """
        From a 1D array of probability predictions, calculates a 2D array of binary predictions with each row
        corresponding to the predictions given one of the passed probability thresholds.

        :param probability_predictions: The array of predicted probabilities for the binary labels.
        :param thresholds: The thresholds to generate binary labels on from the probabilities.
        :return: The array containing the binary labels for each threshold.
        """
        return probability_predictions > thresholds[:, np.newaxis]
