"""Configuration for the pytest tests."""

import pytest
import matplotlib


matplotlib.use('Agg')  # Use non-interactive backend to prevent loss of focus during test.


def pytest_addoption(parser):
    """Adds additional options to the pytest commandline."""
    parser.addoption(
        '--exclude-functional', action='store_true', default=False, help='Run functional tests'
    )
    parser.addoption(
        '--exclude-external', action='store_true', default=False, help='Run tests that rely on external resources'
    )
    parser.addoption(
        '--exclude-slow', action='store_true', default=False, help='Run slow tests'
    )


def pytest_configure(config):
    """Additional configuration options for pytest."""
    config.addinivalue_line(
        'markers', 'functional: Mark a test as a functional test.'
    )
    config.addinivalue_line(
        'markers', 'slow: Mark a test as a slow test.'
    )
    config.addinivalue_line(
        'markers', 'External: Mark a test as requiring an external resource (e.g. makes a network call to a URL).'
    )


def pytest_collection_modifyitems(config, items):
    """Updates the collections based on the passed arguments to pytest."""
    functional_skip_mark = pytest.mark.skip(reason='Options to exclude functional tests were passed.')
    slow_skip_mark = pytest.mark.skip(reason='Options to exclude slow tests were passed.')
    for item in items:
        if 'functional' in item.keywords and config.getoption('--exclude-functional'):
            item.add_marker(functional_skip_mark)
        if 'slow' in item.keywords and config.getoption('--exclude-slow'):
            item.add_marker(slow_skip_mark)
        if 'external' in item.keywords and config.getoption('--exclude-external'):
            item.add_marker(slow_skip_mark)
