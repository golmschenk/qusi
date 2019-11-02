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
