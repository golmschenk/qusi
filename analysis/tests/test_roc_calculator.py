"""Tests for the RocCalculator class."""
import pytest
import numpy as np

from analysis.roc_calculator import RocCalculator


class TestRocCalculator:
    """Tests for the RocCalculator class."""

    @pytest.fixture
    def roc_calculator(self) -> RocCalculator:
        """
        Sets up the ROC calculator for use in a test.

        :return: The calculator.
        """
        return RocCalculator()

    @pytest.fixture
    def label(self) -> np.ndarray:
        """
        A label to use in the tests.

        :return: The label.
        """
        return np.array([1, 0, 0, 0, 1, 1], dtype=np.float32)

    @pytest.fixture
    def prediction(self) -> np.ndarray:
        """
        A prediction to use in the tests.

        :return: The prediction.
        """
        return np.array([0.9, 0.1, 0.4, 0.6, 0.4, 0.8], dtype=np.float32)

    def test_can_generate_thresholded_predictions(self, roc_calculator):
        probability_predictions = np.array([0.9, 0.1, 0.4, 0.6, 0.4, 0.8], dtype=np.float32)
        thresholds = np.array([0.0, 0.5, 1.0])
        expected_thresholded_predictions = np.array([[1, 1, 1, 1, 1, 1],
                                                     [1, 0, 0, 1, 0, 1],
                                                     [0, 0, 0, 0, 0, 0]], dtype=np.bool)
        thresholded_predictions = roc_calculator.threshold_predictions(probability_predictions, thresholds)
        assert np.array_equal(thresholded_predictions, expected_thresholded_predictions)

    def test_can_calculate_confusion_matrix_counts_for_each_prediction_threshold(self, roc_calculator):
        label = np.array([1, 0, 0, 0, 1, 1], dtype=np.bool)
        thresholded_predictions = np.array([[1, 1, 1, 1, 1, 1],
                                            [1, 0, 0, 1, 0, 1],
                                            [0, 0, 0, 0, 0, 0]], dtype=np.bool)
        expected_true_positive_counts = np.array([3, 2, 0], dtype=np.int32)
        expected_false_positive_counts = np.array([3, 1, 0], dtype=np.int32)
        expected_true_negative_counts = np.array([0, 2, 3], dtype=np.int32)
        expected_false_negative_counts = np.array([0, 1, 3], dtype=np.int32)
        confusion_counts = roc_calculator.calculate_confusion_matrix_counts(label, thresholded_predictions)
        true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts = confusion_counts
        assert np.array_equal(true_positive_counts, expected_true_positive_counts)
        assert np.array_equal(false_positive_counts, expected_false_positive_counts)
        assert np.array_equal(true_negative_counts, expected_true_negative_counts)
        assert np.array_equal(false_negative_counts, expected_false_negative_counts)
