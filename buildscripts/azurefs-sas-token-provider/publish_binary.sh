#!/bin/bash
set -exo pipefail
# Copied from Iceberg


# Package Setup
eval "$(micromamba shell hook -s posix)"
micromamba activate sas_build


# Build Pakcage
CHANNEL_NAME=${1:-bodo-binary}

echo "********** Publishing to Artifactory **********"

# We always upload to the main channel since it's a manual pipeline and
# the package is not expected to change.
label="main"

cd buildscripts/azurefs-sas-token-provider/conda-recipe/
conda build . -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${BODO_CHANNEL_NAME} -c conda-forge

# Upload to Anaconda
package=`ls $CONDA_PREFIX/conda-bld/noarch/bodo-azurefs-sas-token-provider*.conda`
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
