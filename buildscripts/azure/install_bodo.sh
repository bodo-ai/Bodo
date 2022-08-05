#!/bin/bash
set -exo pipefail

BODO_VERSION=${1:-}
PLATFORM_DEV_RELEASE=${3:-false}
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`

# Deactivate if another script has already activated the env
source deactivate || true

source activate $CONDA_ENV

# ------ Install Bodo -----------
artifactory_channel=`./buildscripts/azure/get_channel.sh $PLATFORM_DEV_RELEASE`

mamba install -y h5py scipy bodo=$BODO_VERSION -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/$artifactory_channel -c conda-forge
