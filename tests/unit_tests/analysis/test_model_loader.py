"""Tests for the model loader code."""
from pathlib import Path

from ramjet.analysis.model_loader import get_latest_log_directory


class TestModelLoader:
    """Tests for the model loader code."""

    def test_can_get_most_recent_log_directory(self):
        logs_directory = Path(__file__).parent.joinpath(
            "test_model_loader_resources/test_logs"
        )
        latest_log_directory = get_latest_log_directory(logs_directory)
        expected_latest_log_directory = Path(__file__).parent.joinpath(
            "test_model_loader_resources/test_logs/baseline 2019-11-08-13-54-37"
        )
        assert latest_log_directory == expected_latest_log_directory
