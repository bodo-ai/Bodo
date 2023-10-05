#!/bin/bash
set -exo pipefail

PLATFORM_DEV_RELEASE=${1:-false}
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`


bodo_artifactory_channel=`./buildscripts/get_channel.sh $PLATFORM_DEV_RELEASE`
echo "bodo_artifactory_channel: $bodo_artifactory_channel"
bodosql_artifactory_channel=`./buildscripts/bodosql/azure/get_channel.sh`
echo "bodosql_artifactory_channel: $bodosql_artifactory_channel"

set +x
source activate $CONDA_ENV
set -x

# --no-update-deps ensures that no dependencies are upgraded.
# If the conda-lock file is out of date, this would cause this installation
# to fail. To fix, we usually need to update the lock-file.
CONDA_INSTALL="mamba install -y --no-update-deps"
BODO_BODOSQL_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`

# Install Bodo first, and then install BodoSQL so we don't install Bodo from the wrong
# channel if they differ
$CONDA_INSTALL -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/$bodo_artifactory_channel -c conda-forge bodo=${BODO_BODOSQL_VERSION}

# Finally, install BodoSQL
$CONDA_INSTALL -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${bodosql_artifactory_channel} -c conda-forge bodosql=${BODO_BODOSQL_VERSION}
