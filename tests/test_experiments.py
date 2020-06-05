"""
Tests for the experiments module.
"""
from ramjet.experiments import Experiment


class TestExperiment:
    def test_database_attribute_is_abstract(self):
        # noinspection PyUnresolvedReferences
        assert 'database' in Experiment.__abstractmethods__

    def test_model_attribute_is_abstract(self):
        # noinspection PyUnresolvedReferences
        assert 'database' in Experiment.__abstractmethods__
