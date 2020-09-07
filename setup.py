#!/usr/bin/env python
import os
import setuptools

from ea_ml import __project__, __version__, CLI, DESCRIPTION

if os.path.exists('README.md'):
    README = open('README.md').read()
else:
    README = ""
CHANGES = open('CHANGELOG.md').read()

setuptools.setup(
    name=__project__,
    version=__version__,
    description=DESCRIPTION,
    author='Dillon Shapiro',
    author_email='drshapir@bcm.edu',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': [f'{CLI} = ea_ml.cli:main']},
    long_description=f'{README}\n{CHANGES}',
    install_requires=[
        'javabridge>=1.0.18',
        'joblib>=0.16.0',
        'matplotlib>=3.1.1',
        'numpy>=1.16.2',
        'pandas>=0.24.1',
        'pysam>=0.15.2',
        'python-weka-wrapper3>=0.1.7',
        'scikit-learn>=0.20.3',
        'scipy>=1.2.1',
        'seaborn>=0.9.0',
        'statsmodels>=0.10.1',
        'tqdm>=4.48.2'
    ],
    include_package_data=True
)
