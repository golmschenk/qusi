"""
Script to build the Sphinx HTML documentation.
"""

from pathlib import Path
from sphinx.application import Sphinx


docs_directory = Path('.').absolute()
source_directory = docs_directory
configuration_directory = docs_directory
build_directory = Path(configuration_directory, '_build')
doctree_directory = Path(build_directory, '.doctrees')
builder = 'html'

app = Sphinx(source_directory, configuration_directory, build_directory, doctree_directory, builder, freshenv=True)
app.build()
