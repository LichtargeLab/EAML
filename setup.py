#!/usr/bin/env python
import os
import setuptools

from ea_ml import NAME, __version__, CLI, DESCRIPTION

if os.path.exists('README.md'):
    README = open('README.md').read()
else:
    README = ""
CHANGES = open('CHANGELOG.md').read()

setuptools.setup(
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    author='Dillon Shapiro',
    author_email='drshapir@bcm.edu',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': [f'{CLI} = ea_ml.cli:main']},
    long_description=f'{README}\n{CHANGES}',
    install_requires=open('requirements.txt').readlines(),
    include_package_data=True
)
