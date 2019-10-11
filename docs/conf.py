# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'RAMjET'
copyright = '2019, Greg Olmschenk'
author = 'Greg Olmschenk'
master_doc = 'index'


# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'autoapi.extension']

autoapi_dirs = ['..']
autoapi_ignore = ['*envs/*', '*venv/*', '*.pytest_cache/*', '*logs/*', '*data/*', '*docs/*', '*tests/*']

templates_path = ['_templates']
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
