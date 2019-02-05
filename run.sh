#!/usr/bin/env bash

# check for conda and install if needed. Will download python 3.6. Also, could switch this to miniconda
if ! which conda > /dev/null; then
    echo -e "Conda not found! Install? (y/n) \c"
    read REPLY
    if [ "$REPLY" = "y" ]; then
        wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh
        bash ~/anaconda.sh -b -p $HOME/anaconda3
        echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> $HOME/.bashrc # add anaconda bin to the environment
        export PATH="$HOME/anaconda3/bin:$PATH"
    else
        echo "Conda is required for proper virtual environment setup."
        exit 1
    fi
fi

# Set-up the Environment
ENV_NAME='pyEA-ML'

ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *$ENV_NAME* ]]; then
   source activate ${ENV_NAME}
else
    # make virtual environment
    conda create --no-deps -n ${ENV_NAME} --yes
    source activate ${ENV_NAME}
    pip install pip==18.1
    pip install -r requirements.txt
fi

# Set-up .env
if [ ! -f ./.env ]; then
    # Make the file
    touch ./.env
    # Record the results folder destination
    while getopts ":he:d:s:g:" opt; do
        case ${opt} in
            h)
                echo "Usage:"
                echo "./run.sh -e <experiment folder> -d <data> -s <sample file> -g <gene list>"
                echo "./run.sh -h           Display this help message."
                exit 0;;
            e) dotenv -f .env set EXPDIR $OPTARG;;
            d) dotenv -f .env set DATA $OPTARG;;
            s) dotenv -f .env set SAMPLES $OPTARG;;
            g) dotenv -f .env set GENELIST $OPTARG;;
            \?)
                echo "Invalid Option: -$OPTARG" 1>&2
                exit 1;;
        esac
    done
    shift $((OPTIND -1))
fi

# run pipeline
# readlink -f doesn't work on OS X, so this can only be done on a Linux system
repSource=$(readlink -f `dirname ${BASH_SOURCE[0]}`)
python ${repSource}/src/main.py
