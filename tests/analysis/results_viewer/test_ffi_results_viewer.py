from unittest.mock import Mock, patch
import pytest

from ramjet.analysis.results_viewer.ffi_results_viewer import FfiResultsViewer
from ramjet.analysis.results_viewer.results_viewer import ResultsViewer


class TestFfiResultsViewer:
    def test_the_viewer_is_a_subclass_of_the_default_results_viewer(self):
        assert issubclass(FfiResultsViewer, ResultsViewer)
