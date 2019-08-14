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
        echo 'export PATH=/lab/cedar/shared/anaconda3/bin:$PATH' >> $HOME/.bashrc
    else
        echo "Conda is required for proper virtual environment setup."
        exit 1
    fi
fi
source $HOME/.bashrc

# Set-up the Environment
ENV_NAME='pyEA-ML'

ENVS=$(conda env list | awk '{print $1}' )
if [[ ${ENVS} = *${ENV_NAME}* ]]; then
   source activate ${ENV_NAME}
else
    # make virtual environment
    conda env create -f ${repSource}/environment.yml
    source activate ${ENV_NAME}
fi
export JAVA_HOME=${CONDA_PREFIX}/jre
export PATH=${JAVA_HOME}/bin:$PATH

USAGE="Usage:  pyEA-ML/run.sh [-h] <required> <optional>

        Required arguments:
           -e <directory>       experiment directory
           -d <file>            VCF or .npz matrix
           -s <file>            two-column CSV with sample IDs and disease status
           -g <file>            single-column list of genes

        Optional arguments:
           -t <int>             number of threads for Weka to use
           -r <int>             random seed for KFold sampling
           -k <int>             number of KFold samples
           -h                   Display this help message.
       "

# parse command line arguments
seed=111
threads=1
cv=10
while getopts ":he:d:s:g:t:r:k:" opt; do
    case ${opt} in
        h) echo "${USAGE}"
           exit 0;;
        e) exp_dir=$OPTARG;;
        d) data=$OPTARG;;
        s) samples=$OPTARG;;
        g) genelist=$OPTARG;;
        t) threads=$OPTARG;;
        r) seed=$OPTARG;;
        k) cv=$OPTARG;;
        \?) echo "Invalid Option: -$OPTARG" 1>&2
            echo "${USAGE}"
            exit 1;;
        *) echo "${USAGE}"
           exit 1;;
    esac
done
shift $((OPTIND -1))

# run pipeline
cd ${exp_dir}
python ${repSource}/src/main.py ${exp_dir} ${data} ${samples} ${genelist} -t ${threads} -r ${seed} -k ${cv}
