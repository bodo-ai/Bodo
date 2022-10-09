#!/bin/bash
set -exo pipefail
# Copied from Iceberg


# Package Setup
eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate
micromamba install -q -y boa anaconda-client conda-verify curl -c conda-forge


# Build Pakcage
CHANNEL_NAME=${1:-bodo-binary}

echo "********** Publishing to Artifactory **********"
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`
ANACONDA_TOKEN=`cat $HOME/secret_file | grep anaconda.org.token | cut -f 2 -d' '`

# We always upload to the main channel since it's a manual pipeline and
# the package is not expected to change.
label="main"

cd buildscripts/azurefs-sas-token-provider/conda-recipe/
conda mambabuild . --no-test -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${BODO_CHANNEL_NAME} -c conda-forge

# Upload to Anaconda
package=`ls $CONDA_PREFIX/conda-bld/noarch/bodo-azurefs-sas-token-provider*.tar.bz2`
if [[ -z "$package" ]]; then
  echo "Unable to Find Package. Exiting ..."
  exit 1
fi

package_name=`basename $package`
echo "Package Name: $package_name"
curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/noarch/$package_name"
if [[ ! -z "$label" ]]; then
    anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --force
fi

# Reindex Conda
curl -s -X POST "https://$USERNAME:$TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex?async=0"
