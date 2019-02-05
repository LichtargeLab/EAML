===============================
pyEA-ML
===============================

* Free software: ISC license

The Python version of the lab EA-ML pipeline.

This pipeline uses a series of supervised learning algorithms to score a gene's contribution to disease risk based on it's case/control separation ability. This is all based on a probability score (pAFF) computed using EA.

The output is a ranked list of genes which can be used to inform followup analyses.

Usage
----------------------

The run.sh script will install the required virtual environment and dependencies.
To run the pipeline from start to finish::
    cd <experiment folder>
    chmod +x pyEA-ML/run.sh
    pyEA-ML/run.sh -e <experiment folder> -d <data> -s <sample file> -g <gene list>

If a .env file already exists within your experiment folder, no input arguments are necessary as long as you run from the experiment folder.

Expanded Usage
####################

If you wish to use the pipeline modularly, then create an environment with all of the dependencies using the environment.yml file::

    # Use conda to make the environment
    conda env create -f environment.yml

The alternative is to use virtual environment::

    # install virtualenv if needed
    sudo apt install virtualenv
    virtualenv -p python3 pyEA-ML
    source pyEA-ML/bin/activate
    pip install -r requirements.txt

Import module functions/classes::

    import sys
    sys.path.append('monorepo/pipelines/pyEA-ML')

Input Requirements
----------------------

In order to use this pipeline properly it requires 3 input files:

    1. A VCF file containing all cohort variants annotated with gene information, EA scores, and genotype information for each cohort sample.
    2. A comma-delimited list of samples along with their disease status (0 or 1).
    3. A single column list of genes to test.

The VCF file should follow proper formatting as described `here <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`_. Additionally, some extra information is required:
    * 'gene' and 'EA' annotations as fields in the INFO column
    * Different fields in INFO and FORMAT columns should be defined in the header, with type information
    * 'EA' attribute must be typed as a 'String' (Type=String), this is because of the way EA labels variants by transcript, and other variant annotation softwares do this on a variant-wise basis instead

Software Requirements
----------------------

* Java Runtime Environment (build 1.8.0_181-b13)
* Anaconda3
* Python 3.6.8

Credits
----------------------

Tools used in rendering this package:

*  Cookiecutter_
*  `cookiecutter-pypackage`_

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
