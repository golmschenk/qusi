"""
Tests for the experiments module.
"""
from ramjet.experiments import Experiment


class TestExperiment:
    def test_database_attribute_exists(self):
        # noinspection PyUnresolvedReferences
        assert 'database' in Experiment.__dict__.keys()

    def test_model_attribute_exists(self):
        # noinspection PyUnresolvedReferences
        assert 'model' in Experiment.__dict__.keys()

    def test_run_name_attribute_exists(self):
        # noinspection PyUnresolvedReferences
        assert 'run_name' in Experiment.__dict__.keys()
