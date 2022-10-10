import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='astroramjet',
    version='0.6.1',
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    url='https://github.com/golmschenk/ramjet',
    license='Apache License 2.0',
    author='Greg Olmschenk',
    author_email='golmschenk@gmail.com',
    description='',
    long_description=long_description,
    install_requires=['retrying', 'numpy', 'pandas', 'tensorflow', 'astropy', 'astroquery', 'requests', 'pyarrow',
                      'matplotlib', 'pipreqs', 'bokeh', 'pymc3', 'Theano', 'exoplanet', 'scipy', 'setuptools',
                      'dataset', 'pathos', 'peewee']
)
