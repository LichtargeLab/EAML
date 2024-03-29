#!/usr/bin/env python
import setuptools

from eaml import __project__, __version__, CLI, DESCRIPTION

README = open('README.md').read()
CHANGES = open('CHANGELOG.md').read()

setuptools.setup(
    name=__project__,
    version=__version__,
    description=DESCRIPTION,
    long_description=f'{README}\n{CHANGES}',
    author='Dillon Shapiro',
    author_email='drshapir@bcm.edu',
    url='https://github.com/LichtargeLab/EAML',
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points={'console_scripts': [f'{CLI} = eaml.cli:main', 'ea-ml = eaml.cli:main']},
    install_requires=[
        'adjusttext',
        'joblib',
        'matplotlib>=3.1.1',
        'numpy>=1.16.2',
        'pandas>=0.24.1',
        'pysam>=0.15.2',
        'scikit-learn>=0.24.2',
        'scipy>=1.2.1',
        'seaborn>=0.9.0',
        'statsmodels>=0.10.1',
        'tqdm'
    ]
)
