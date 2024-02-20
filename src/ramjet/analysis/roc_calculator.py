"""
Code for a class to calculate receiver operating characteristic (ROC) curves.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class RocCalculator:
    """
    A class to calculate receiver operating characteristic (ROC) curves.
    """

    def __init__(self):
        self.true_positive_counts: int = 0
        self.false_positive_counts: int = 0
        self.true_negative_counts: int = 0
        self.false_negative_counts: int = 0
        self.thresholds: np.ndarray = np.linspace(0, 1, 1000)

    @property
    def true_positive_rates(self) -> np.ndarray:
        """
        Calculates the true positive rates for the accumulated confusion matrix counts for each threshold.

        :return: The true positive rates for each threshold.
        """
        return self.true_positive_counts / (self.true_positive_counts + self.false_negative_counts)

    @property
    def false_positive_rates(self) -> np.ndarray:
        """
        Calculates the false positive rates for the accumulated confusion matrix counts for each threshold.

        :return: The false positive rates for each threshold.
        """
        return self.false_positive_counts / (self.true_negative_counts + self.false_positive_counts)

    @staticmethod
    def threshold_predictions(probability_predictions: np.ndarray, thresholds: np.ndarray):
        """
        From a 1D array of probability predictions, calculates a 2D array of binary predictions with each row
        corresponding to the predictions given one of the passed probability thresholds.

        :param probability_predictions: The array of predicted probabilities for the binary labels.
        :param thresholds: The thresholds to generate binary labels on from the probabilities.
        :return: The array containing the binary labels for each threshold.
        """
        return probability_predictions > thresholds[:, np.newaxis]

    @staticmethod
    def calculate_confusion_matrix_counts(label: np.ndarray, predictions: np.ndarray):
        """
        Calculates the confusion matrix counts for a 1D set of true binary labels a 2D array of predictions, where
        each row corresponds to a prediction to compare.

        :param label: A 1D binary array label.
        :param predictions: A 2D array of predictions, each row of which is to be compared to the label.
        :return: The confusion matrix values for each row of the predictions.
        """
        true_positives = np.count_nonzero(label & predictions, axis=1)
        false_positives = np.count_nonzero(~label & predictions, axis=1)
        true_negatives = np.count_nonzero(~label & ~predictions, axis=1)
        false_negatives = np.count_nonzero(label & ~predictions, axis=1)
        return true_positives, false_positives, true_negatives, false_negatives

    def accumulate_confusion_matrix_counts(self, label: np.ndarray, prediction: np.ndarray):
        """
        Calculates the confusion matrix counts for a given label and probability prediction pair, and adds those counts
        to the totals.

        :param label: The 1D array binary label.
        :param prediction: The 1D probability array prediction.
        """
        thresholded_predictions = self.threshold_predictions(prediction, self.thresholds)
        counts = self.calculate_confusion_matrix_counts(label, thresholded_predictions)
        true_positives, false_positives, true_negatives, false_negatives = counts
        self.true_positive_counts += true_positives
        self.false_positive_counts += false_positives
        self.true_negative_counts += true_negatives
        self.false_negative_counts += false_negatives

    def generate_roc_plot(self, output_path: str | Path = "roc_curve.svg", title: str | None = None):
        """
        Generates a ROC curve plot from the confusion matrix totals which have been accumulated.

        :param output_path: The path to save the plot image file to.
        :param title: The title to add to the plot.
        """
        with plt.style.context("seaborn-darkgrid"):
            figure, axes = plt.subplots()
            axes.plot(self.false_positive_rates, self.true_positive_rates)
            axes.set_xlabel("False positive rate")
            axes.set_ylabel("True positive rate")
            if title is not None:
                axes.set_title(title)
            figure.patch.set_alpha(0)  # Transparent figure background while keeping grid background.
            # Need to explicitly state face color or the save will override it.
            plt.savefig(output_path, facecolor=figure.get_facecolor())
