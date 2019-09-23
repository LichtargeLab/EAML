#!/usr/bin/env bash

### Environment Setup ###
# readlink -f doesn't work on OS X, so this can only be done on a Linux system
# Alternatively, this will work on OS X if `greadlink` is installed through
# Homebrew and added to PATH
repSource=$(readlink -f `dirname ${BASH_SOURCE[0]}`)
# check for conda
if ! which conda > /dev/null; then
    echo -e "Conda not found! Install? (y/n) \c"
    read REPLY
    if [[ "$REPLY" = "y" ]]; then
        echo -e "Are you on a lab server? (y/n) \c"
        read REPLY
        if [[ "$REPLY" = "y" ]]; then
            echo 'export PATH=/lab/cedar/shared/anaconda3/bin:$PATH' >> $HOME/.bashrc
        else
            wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
            bash ~/miniconda.sh -b -p $HOME/miniconda3
        fi
    else
        echo "Conda is required for proper virtual environment setup. Please install Conda on your system."
        exit 1
    fi
fi
source $HOME/.bashrc

# Set-up the Environment
ENV_NAME='pyEA-ML'

ENVS=$(conda env list | awk '{print $1}' )
if [[ ${ENVS} != *${ENV_NAME}* ]]; then
   # make virtual environment
    conda create -n ${ENV_NAME} -c bioconda python=3.7.2 java-jdk=8.0.92
    source activate ${ENV_NAME}
    export JAVA_HOME=${CONDA_PREFIX}/jre
    conda env update -f ${repSource}/environment.yml
else
    echo "pyEA-ML environment already installed."
fi
