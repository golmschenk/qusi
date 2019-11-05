"""
Configuration file for the Sphinx documentation builder.
"""
import inspect
import sys
import os
import datetime
from typing import Dict
from importlib import import_module
from autoapi.mappers.python.objects import PythonPythonMapper
from sphinx.application import Sphinx
from git import Repo, Head, Tag

# Path setup

sys.path.insert(0, os.path.abspath('..'))

# Project information

project = 'RAMjET'
copyright = f'{datetime.datetime.now().year}, Greg Olmschenk'
author = 'Greg Olmschenk'
master_doc = 'index'

# General Sphinx configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'autoapi.extension', 'sphinx.ext.linkcode']

add_module_names = False
templates_path = ['_templates']
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']

# AutoAPI configuration.

autoapi_dirs = ['..']
autoapi_ignore = ['*envs/*', '*venv/*', '*.pytest_cache/*', '*logs/*', '*data/*', '*docs/conf.py', '*docs/build.py',
                  '*tests/*', '*conftest.py', '*train.py', '*infer.py']
autoapi_template_dir = '_templates/autoapi'


# noinspection PyUnusedLocal
def autoapi_skip_member(app: Sphinx, what: str, name: str, obj: PythonPythonMapper, would_skip: bool, options: Dict):
    """Project specific skip function for AutoAPI."""
    if '__init__' in name:
        return False
    return would_skip


# Source code linking.

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to a Python object.
    """

    class PlaceholderException(Exception):
        """A placeholder exception meant to show where to add specific exceptions in the future for the code linking."""
        pass

    if domain != 'py':
        return None
    module_name = info['module']
    full_name = info['fullname']
    import_module(module_name)
    module = sys.modules.get(module_name)
    if module is None:
        return None
    object_ = module
    for part in full_name.split('.'):
        try:
            object_ = getattr(object_, part)
        except AttributeError:
            return None
    try:
        file_name = inspect.getsourcefile(object_)
    except PlaceholderException:
        file_name = None
    if not file_name:
        return None
    file_name = os.path.relpath(file_name, start=os.path.abspath('..'))
    try:
        source, line_number = inspect.getsourcelines(object_)
    except PlaceholderException:
        line_number = None
        source = None
    if line_number is not None and source is not None:
        line_range_jump_option = f'#L{line_number}-L{line_number + len(source) - 1}'
    else:
        line_range_jump_option = ''
    repository = Repo('..')
    matching_refs = []
    for ref in list(repository.refs):  # Manually iterate over refs, because ReadTheDocs uses a detached head.
        if ref.commit == repository.head.commit:
            matching_refs.append(ref)
    head_ref_names = [ref.name.split('/')[-1] for ref in matching_refs if isinstance(ref, Head)]
    tag_ref_names = [ref.name.split('/')[-1] for ref in matching_refs if isinstance(ref, Tag)]
    if 'master' in head_ref_names:  # First check if the commit matches the master branch.
        ref_name = 'master'
    elif tag_ref_names:  # Second, check if the commit matches a tag.
        ref_name = tag_ref_names[0]
    elif head_ref_names:  # Third, check if the commit matches a branch.
        ref_name = head_ref_names[0]
    else:  # If none of the above are true, have the doc refer to the specific commit.
        ref_name = repository.head.commit
    return f'http://github.com/golmschenk/ramjet/blob/{ref_name}/{file_name}{line_range_jump_option}'


# HTML output configuration

html_theme = 'press'
html_static_path = ['_static']


def setup(app: Sphinx):
    """Project specific setup for Sphinx."""
    app.connect('autoapi-skip-member', autoapi_skip_member)
