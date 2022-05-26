#!/bin/bash
# Copied from BodoSQL
set -xeo pipefail

export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-10.15}
export CONDA_BUILD_SYSROOT="/opt/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk"
if [[ $TARGET_NAME = "macOS_ARM" ]]; then
    export PATH=$HOME/miniforge3/bin:$PATH
else
    export PATH=$HOME/miniconda3/bin:$PATH
fi
source activate build_iceberg_connector

# Upload to artifactory
conda install -y conda-build anaconda-client conda-verify curl -c conda-forge

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}

echo "********** Publishing to Artifactory **********"
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`
ANACONDA_TOKEN=`cat $HOME/secret_file | grep anaconda.org.token | cut -f 2 -d' '`

# Get the Connector Version
export CONNECTOR_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`
export IS_RELEASE=`git tag --points-at HEAD`

# We follow the following convention for release:
# Major Release: Original month release, i.e. 2021.8beta or 2021.11
# There should always be 1 dot, even in beta.
# Minor Release: Repeat release in a month, there should be a second dot
# i.e. 2021.8.1beta or 2021.11.2
# Release Candidate: If we need to test a release before the major release
# we will create a minor release ending in .0 and append rc#
# i.e. 2021.9.0betarc1 or 2021.11.0rc2
IS_MAJOR_RELEASE=0
if [[ -n "$IS_RELEASE" ]]; then
    IS_MAJOR_RELEASE=`python -c "import os; print(int(len(os.environ[\"CONNECTOR_VERSION\"].split(\".\")) < 3))"`
fi
label=""
if [[ "$IS_MAJOR_RELEASE" == 1 ]] && [[ "$CHANNEL_NAME" == "bodo.ai" ]]; then
    # If we have a major release upload with our main anaconda label
    label="main"
elif [[ "$CHANNEL_NAME" == "bodo.ai" ]] && [[ -n "$IS_RELEASE" ]]; then
    # If we have a minor release upload with our dev anaconda label
    label="dev"
fi

if [[ "$TARGET_NAME" == "macOS_ARM" ]]; then
    cd buildscripts/iceberg/m1-conda-recipe/
else
    cd buildscripts/iceberg/conda-recipe/
fi
conda-build . -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${BODO_CHANNEL_NAME} -c conda-forge --no-test

for package in `ls $CONDA_PREFIX/conda-bld/$OS_DIR/bodo-iceberg-connector*.tar.bz2`; do
    package_name=`basename $package`
    curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/${OS_DIR}/$package_name"
    if [[ ! -z "$label" ]]; then
        anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label
    fi
done

# reindex conda
curl -X POST https://$USERNAME:$TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex
