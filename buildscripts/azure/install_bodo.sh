#!/bin/bash
set -exo pipefail

USE_NUMBA_DEV=${1:-false}
BODO_VERSION=${2:-}
PLATFORM_DEV_RELEASE=${3:-false}
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`

source activate $CONDA_ENV

# ------ Install Bodo -----------
artifactory_channel=`./buildscripts/azure/get_channel.sh $USE_NUMBA_DEV $PLATFORM_DEV_RELEASE`
numba_channel_flag=""

if [[ "$USE_NUMBA_DEV" == "true" ]]; then
    echo "Init numba_channel_flag variable"
    numba_channel_flag="-c numba/label/dev"
fi

conda install -y h5py=2.10 scipy bodo=$BODO_VERSION $numba_channel_flag -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/$artifactory_channel/$sub_channel -c numba -c conda-forge
