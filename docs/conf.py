"""
Configuration file for the Sphinx documentation builder.
"""

import sys
import os
import datetime
from typing import Dict

from autoapi.mappers.python.objects import PythonPythonMapper
from sphinx.application import Sphinx


# Path setup

sys.path.insert(0, os.path.abspath('..'))


# Project information

project = 'RAMjET'
copyright = f'{datetime.datetime.now().year}, Greg Olmschenk'
author = 'Greg Olmschenk'
master_doc = 'index'


# General Sphinx configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'autoapi.extension']

add_module_names = False
templates_path = ['_templates']
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']


# AutoAPI configuration.

autoapi_dirs = ['..']
autoapi_ignore = ['*envs/*', '*venv/*', '*.pytest_cache/*', '*logs/*', '*data/*', '*docs/conf.py', '*tests/*']
autoapi_template_dir = '_templates/autoapi'


# HTML output configuration

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# noinspection PyUnusedLocal
def autoapi_skip_member(app: Sphinx, what: str, name: str, obj: PythonPythonMapper, would_skip: bool, options: Dict):
    """Project specific skip function for AutoAPI."""
    if '__init__' in name:
        return False
    return would_skip


def setup(app: Sphinx):
    """Project specific setup for Sphinx."""
    app.connect('autoapi-skip-member', autoapi_skip_member)
