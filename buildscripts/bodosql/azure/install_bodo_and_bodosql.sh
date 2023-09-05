#!/bin/bash
set -exo pipefail

PLATFORM_DEV_RELEASE=${1:-false}
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`


bodo_artifactory_channel=`./buildscripts/get_channel.sh $PLATFORM_DEV_RELEASE`
echo "bodo_artifactory_channel: $bodo_artifactory_channel"
bodosql_artifactory_channel=`./buildscripts/bodosql/azure/get_channel.sh`
echo "bodosql_artifactory_channel: $bodosql_artifactory_channel"


export PATH=$HOME/mambaforge/bin:${PATH}
source activate $CONDA_ENV

CONDA_INSTALL="mamba install -y"
BODO_BODOSQL_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`

# Install Bodo first, followed by the iceberg connector, and then install BodoSQL so we don't install Bodo from the wrong
# channel if they differ
$CONDA_INSTALL -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/$bodo_artifactory_channel -c conda-forge bodo=${BODO_BODOSQL_VERSION}
# TODO: figure out how to version lock the iceberg connector in the same way that we do the bodo version
# Iceberg and BodoSQL upload to the same channel whether it's a release or not.
$CONDA_INSTALL -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/$bodosql_artifactory_channel -c conda-forge bodo-iceberg-connector
# Install sqlalchemy, the snowflake connector, and snowflake-sqlalchemy, which is needed as a testing dependency for snowflake
$CONDA_INSTALL -c conda-forge sqlalchemy snowflake-sqlalchemy snowflake-connector-python
# Finally, install bodosql
$CONDA_INSTALL bodosql=${BODO_BODOSQL_VERSION}  -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${bodosql_artifactory_channel} -c conda-forge
