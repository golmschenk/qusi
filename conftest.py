"""Configuration for the pytest tests."""

import pytest


collect_ignore = ['envs/', 'venv/', 'data/', 'logs/']


def pytest_addoption(parser):
    """Adds additional options to the pytest commandline."""
    parser.addoption(
        "--run-functional", action="store_true", default=False, help="Run functional tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "functional: Mark a test as a functional test."
    )


def pytest_collection_modifyitems(config, items):
    """Updates the collections based on the passed arguments to pytest."""
    if config.getoption("--run-functional"):
        return  # If options is passed, don't skip the functional tests.
    functional_skip_mark = pytest.mark.skip(reason="Needs `--run-functional` to run functional tests.")
    for item in items:
        if "functional" in item.keywords:
            item.add_marker(functional_skip_mark)
