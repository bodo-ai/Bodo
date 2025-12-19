#!/bin/bash
set -exo pipefail
# Copied from BodoSQL with micromamba changes


# Package Setup
eval "$(micromamba shell hook -s posix)"
micromamba activate iceberg_build


# Build Pakcage
CHANNEL_NAME=${1:-bodo-binary}

echo "********** Publishing to Artifactory **********"
# Get the Connector Version
export CONNECTOR_VERSION=`python -m setuptools_scm`
export IS_RELEASE=`git tag --points-at HEAD`

# We follow the following convention for release:
# Major Release: Original month release, i.e. 2021.8b1 or 2021.11
# There should always be 1 dot, even in beta.
# Minor Release: Repeat release in a month, there should be a second dot
# i.e. 2021.8.1beta or 2021.11.2
# Release Candidate: If we need to test a release before the major release
# we will create a minor release ending in .0 and append rc#
# i.e. 2021.9.0betarc1 or 2021.11.0rc2
# For more information, please see our confluence doc: https://bodo.atlassian.net/wiki/spaces/B/pages/1020592198/Release+Checklist
label=""
if [[ -n "$IS_RELEASE" ]] && [[ "$CHANNEL_NAME" == "bodo.ai" ]]; then
    # If we have a major release upload with our main anaconda label
    label="main"
fi

cd buildscripts/iceberg/conda-recipe/
conda build . -c conda-forge

# Upload to Anaconda
package=`ls $CONDA_PREFIX/conda-bld/noarch/bodo-iceberg-connector*.conda`
if [[ -z "$package" ]]; then
  echo "Unable to Find Package. Exiting ..."
  exit 1
fi

package_name=`basename $package`
echo "Package Name: $package_name"
if [[ ! -z "$label" ]]; then
    anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --skip-existing
fi

