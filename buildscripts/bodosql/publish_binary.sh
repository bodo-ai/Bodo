#!/bin/bash
set -exo pipefail

# Package Setup
eval "$(micromamba shell hook -s posix)"
micromamba activate bodosql_build

BODOSQL_CHANNEL_NAME=${1:-bodo-binary}

echo "********** Publishing to Artifactory **********"
# Get the BodoSQL version
# Since we build BodoSQL after Bodo on Azure, we can tie the BodoSQL and Bodo version together
export BODOSQL_VERSION=`python -m setuptools_scm`
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
label="main"
PACKAGE_DIR=$HOME/conda-bld

# Upload to Anaconda
package=`ls $PACKAGE_DIR/noarch/bodosql*.conda`
if [[ -z "$package" ]]; then
  echo "Unable to Find Package. Exiting ..."
  exit 1
fi

package_name=`basename $package`
echo "Package Name: $package_name"
if [[ ! -z "$label" ]]; then
    # `--skip-existing` skips the upload in case the package already exists.
    # Since the pipeline runs every night, we don't want to replace
    # the package on anaconda and accidentally break something.
    anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --skip-existing
fi

# Block on checking if the reindex has failed.
set +e
exit_status=1
while [[ $exit_status != 0 ]]
do
    sleep 30
    conda search bodosql="${BODOSQL_VERSION}" -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${BODOSQL_CHANNEL_NAME}/noarch
    exit_status=$?
done
