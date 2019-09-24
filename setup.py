#!/usr/bin/env python3
"""
Created on 9/20/19

@author: dillonshapiro
"""
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
    install_requires=open('requirements.txt').readlines()
)
