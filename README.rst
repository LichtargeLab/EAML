# pyEA-ML

The Python version of the lab EA-ML pipeline.

* Free software: ISC license

## Usage

if you wish to use Anaconda, then build a wheel and use the environment.yml file:

    # Build a wheel
    python setup.py bdist_wheel
    # Use the wheel to make the environment
    conda env create -f environment.yml

The alternative is to use virtual environment::

    # install virtualenv if needed
    sudo apt install virtualenv

Create an environment::

    virtualenv -p python3 pyEA-ML_env
    source pyEA-ML_env/bin/activate
    pip install -r requirements/dev.txt
    # Install the kernel into IPython
    python -m ipykernel install --user --name pyEA-ML_env --display-name "Python (pyEA-ML_env)"




## Features

* TODO: Fill in Features

## Credits

Tools used in rendering this package:

*  Cookiecutter_
*  `cookiecutter-pypackage`_

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
