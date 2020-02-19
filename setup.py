import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='astroramjet',
    version='0.3.5',
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    url='https://github.com/golmschenk/ramjet',
    license='Apache License 2.0',
    author='Greg Olmschenk',
    author_email='golmschenk@gmail.com',
    description='',
    long_description=long_description,
    install_requires=['retrying', 'numpy', 'pandas', 'tensorflow', 'astropy', 'astroquery', 'requests', 'pytest',
                      'pyarrow', 'Sphinx', 'sphinx-autoapi', 'sphinx-press-theme', 'GitPython', 'matplotlib', 'pipreqs']
)
